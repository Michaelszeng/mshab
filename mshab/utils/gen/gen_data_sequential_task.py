"""
data (.h5 and .json) saved to mshab_exps/gen_data_save_trajectories/{task}/sequential/train/all/

if RECORD_VIDEO is True, video saved to mshab_exps/gen_data_save_trajectories/{task}/sequential/train/all/eval_videos/
"""

import json
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

# ManiSkill specific imports
import mani_skill.envs
import numpy as np
import torch
from dacite import from_dict
from gymnasium import spaces
from mani_skill import ASSET_DIR
from mani_skill import logger as ms_logger
from mani_skill.utils import common
from omegaconf import OmegaConf
from tqdm import tqdm

from mshab.agents.ppo import Agent as PPOAgent
from mshab.agents.sac import Agent as SACAgent
from mshab.envs.make import EnvConfig, make_env
from mshab.envs.planner import CloseSubtask, OpenSubtask, PickSubtask, PlaceSubtask
from mshab.envs.wrappers.record_seq_task import RecordEpisodeSequentialTask
from mshab.utils.array import recursive_slice, to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.logger import Logger, LoggerConfig
from mshab.utils.time import NonOverlappingTimeProfiler

if TYPE_CHECKING:
    from mshab.envs import SequentialTaskEnv

# Note: commenting out "all" policies since object-specific policies are expected to perform better
POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS = dict(
    rl=dict(
        tidy_house=dict(
            pick=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                # "all",
            ],
            place=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                # "all",
            ],
            navigate=["all"],
        ),
        prepare_groceries=dict(
            pick=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                # "all",
            ],
            place=[
                "002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "024_bowl",
                # "all",
            ],
            navigate=["all"],
        ),
        set_table=dict(
            # pick=["013_apple", "024_bowl", "all"],
            # place=["013_apple", "024_bowl", "all"],
            pick=["013_apple", "024_bowl"],
            place=["013_apple", "024_bowl"],
            navigate=["all"],
            open=["fridge", "kitchen_counter"],
            close=["fridge", "kitchen_counter"],
        ),
    ),
)

NUM_ENVS = 1  # Number of environments to run in parallel
SEED = 1
MAX_TRAJECTORIES = 1000  # Number of saved episodes (full sequential task attempts) that pass the filtering criteria

DEMO_FILTER = "success"  # "any" | "success" | "min_success_subtasks"
MIN_SUCCESS_SUBTASKS = 6  # None or int if DEMO_FILTER == "min_success_subtasks"

SAVE_TRAJECTORIES = True
RECORD_VIDEO = True
DEBUG_VIDEO_GEN = False

POLICY_TYPE = "rl_per_obj"
POLICY_KEY = "rl"

human_conf = dict(
    num_envs=1,
    render_mode="human",
    shader_dir="minimal",
    env_kwargs=dict(
        require_build_configs_repeated_equally_across_envs=False,
    ),
)
default_conf = dict(
    num_envs=NUM_ENVS,  # RCAD has 63 train scenes, so 252 envs -> 4 parallel envs reserved for each scene
)

CONF = human_conf


def eval(task, task_plan_path):
    # timer
    timer = NonOverlappingTimeProfiler()

    # seeding
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # NOTE (arth): mps backend on macs not supported since some fns aren't implemented
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------------------------------
    # ENVS
    # -------------------------------------------------------------------------------------------------

    eval_env_cfg = EnvConfig(
        **CONF,
        env_id="SequentialTask-v0",
        obs_mode="rgbd",
        max_episode_steps=dict(
            tidy_house=200 * 10 + 500 * 10,
            prepare_groceries=200 * 6 + 500 * 6,
            set_table=200 * 8 + 500 * 8,
        )[task],
        record_video=RECORD_VIDEO or DEBUG_VIDEO_GEN,
        info_on_video=False,
        debug_video=DEBUG_VIDEO_GEN,
        debug_video_gen=DEBUG_VIDEO_GEN,
        continuous_task=False,
        cat_state=True,
        cat_pixels=False,
        task_plan_fp=(ASSET_DIR / task_plan_path.replace("{task}", task)),
    )
    logger_cfg = LoggerConfig(
        workspace="mshab_exps",
        exp_name=(
            f"gen_data_save_trajectories/{task}/sequential/train/all"
            if SAVE_TRAJECTORIES
            else f"gen_data/{task}/sequential/train/all"
        ),
        clear_out=False,
        tensorboard=False,
        wandb=False,
        exp_cfg=dict(env_cfg=asdict(eval_env_cfg)),
    )
    logger = Logger(
        logger_cfg=logger_cfg,
        save_fn=None,
    )
    wrappers = []
    if SAVE_TRAJECTORIES:
        wrappers = [
            partial(
                RecordEpisodeSequentialTask,
                output_dir=logger.exp_path,
                save_trajectory=True,
                save_video=False,
                info_on_video=False,
                save_on_reset=True,
                save_video_trigger=None,
                max_steps_per_video=None,
                clean_on_close=True,
                record_reward=True,
                source_type="RL",
                source_desc=f"Chained RL policies executing long-horizon {task} task.",
                record_env_state=False,
                max_trajectories=MAX_TRAJECTORIES,
                demo_filter=DEMO_FILTER,
                min_success_subtasks=MIN_SUCCESS_SUBTASKS,
            )
        ]
    eval_envs = make_env(
        eval_env_cfg,
        video_path=logger.eval_video_path,
        wrappers=wrappers,
    )
    uenv: SequentialTaskEnv = eval_envs.unwrapped
    eval_obs, _ = eval_envs.reset(seed=SEED, options=dict(reconfigure=True))
    if uenv.render_mode == "human":
        uenv.render()

        _original_after_control_step = uenv._after_control_step
        _original_after_simulation_step = uenv._after_simulation_step

        time_per_sim_step = uenv.control_timestep / uenv._sim_steps_per_control

        def wrapped_after_control_step(self):
            _original_after_control_step()

            # self._realtime_drift += time.time() - self._control_step_end_time
            # if abs(self._realtime_drift) > 1e-3:
            #     ms_logger.warning(
            #         f"Approx _step_action realtime drift of {self._realtime_drift}"
            #     )

            self._control_step_start_time = time.time()
            self._cur_sim_step = 0
            self._control_step_end_time = self._control_step_start_time + self.control_timestep

        def wrapped_after_simulation_step(self):
            _original_after_simulation_step()
            if getattr(self, "_control_step_start_time", None) is None:
                self._control_step_start_time = time.time()
                self._cur_sim_step = 0
                self._control_step_end_time = self._control_step_start_time + self.control_timestep
                self._realtime_drift = 0

            step_end_time = self._control_step_start_time + (time_per_sim_step * (self._cur_sim_step + 1))
            if time.time() < step_end_time:
                if self.gpu_sim_enabled:
                    self.scene._gpu_fetch_all()
                self.render()
                sleep_time = step_end_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self._cur_sim_step += 1

        uenv._after_control_step = wrapped_after_control_step.__get__(uenv)
        uenv._after_simulation_step = wrapped_after_simulation_step.__get__(uenv)

    # -------------------------------------------------------------------------------------------------
    # SPACES
    # -------------------------------------------------------------------------------------------------

    obs_space = uenv.single_observation_space
    act_space = uenv.single_action_space

    # -------------------------------------------------------------------------------------------------
    # AGENT
    # -------------------------------------------------------------------------------------------------

    dp_action_history = deque([])

    def get_policy_act_fn(algo_cfg_path, algo_ckpt_path):
        algo_cfg = parse_cfg(default_cfg_path=algo_cfg_path).algo
        if algo_cfg.name == "ppo":
            policy = PPOAgent(eval_obs, act_space.shape)
            policy.eval()
            policy.load_state_dict(torch.load(algo_ckpt_path, map_location=device)["agent"])
            policy.to(device)
            policy_act_fn = lambda obs: policy.get_action(obs, deterministic=True)
        elif algo_cfg.name == "sac":
            pixels_obs_space: spaces.Dict = obs_space["pixels"]
            state_obs_space: spaces.Box = obs_space["state"]
            model_pixel_obs_space = dict()
            for k, space in pixels_obs_space.items():
                shape, low, high, dtype = (
                    space.shape,
                    space.low,
                    space.high,
                    space.dtype,
                )
                if len(shape) == 4:
                    shape = (shape[0] * shape[1], shape[-2], shape[-1])
                    low = low.reshape((-1, *low.shape[-2:]))
                    high = high.reshape((-1, *high.shape[-2:]))
                model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
            model_pixel_obs_space = spaces.Dict(model_pixel_obs_space)
            policy = SACAgent(
                model_pixel_obs_space,
                state_obs_space.shape,
                act_space.shape,
                actor_hidden_dims=list(algo_cfg.actor_hidden_dims),
                critic_hidden_dims=list(algo_cfg.critic_hidden_dims),
                critic_layer_norm=algo_cfg.critic_layer_norm,
                critic_dropout=algo_cfg.critic_dropout,
                encoder_pixels_feature_dim=algo_cfg.encoder_pixels_feature_dim,
                encoder_state_feature_dim=algo_cfg.encoder_state_feature_dim,
                cnn_features=list(algo_cfg.cnn_features),
                cnn_filters=list(algo_cfg.cnn_filters),
                cnn_strides=list(algo_cfg.cnn_strides),
                cnn_padding=algo_cfg.cnn_padding,
                log_std_min=algo_cfg.actor_log_std_min,
                log_std_max=algo_cfg.actor_log_std_max,
                device=device,
            )
            policy.eval()
            policy.load_state_dict(torch.load(algo_ckpt_path, map_location=device)["agent"])
            policy.to(device)
            policy_act_fn = lambda obs: policy.actor(
                obs["pixels"],
                obs["state"],
                compute_pi=False,
                compute_log_pi=False,
            )[0]
        else:
            raise NotImplementedError(f"algo {algo_cfg.name} not supported")
        policy_act_fn(to_tensor(eval_obs, device=device, dtype="float"))
        return policy_act_fn

    mshab_ckpt_dir = ASSET_DIR / "mshab_checkpoints"
    if not mshab_ckpt_dir.exists():
        mshab_ckpt_dir = Path("mshab_checkpoints")

    policies = dict()
    for subtask_name, subtask_targs in POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS[POLICY_KEY][task].items():
        policies[subtask_name] = dict()
        for targ_name in subtask_targs:
            cfg_path = mshab_ckpt_dir / POLICY_KEY / task / subtask_name / targ_name / "config.yml"
            ckpt_path = mshab_ckpt_dir / POLICY_KEY / task / subtask_name / targ_name / "policy.pt"
            policies[subtask_name][targ_name] = get_policy_act_fn(cfg_path, ckpt_path)

    def act(obs):
        with torch.no_grad():
            with torch.device(device):
                action = torch.zeros(eval_envs.num_envs, *act_space.shape)

                # get subtask_type for subtask policy querying
                subtask_pointer = uenv.subtask_pointer.clone()
                get_subtask_type = lambda: uenv.task_ids[
                    torch.clip(
                        subtask_pointer,
                        max=len(uenv.task_plan) - 1,
                    )
                ]
                subtask_type = get_subtask_type()

                # find correct envs for each subtask policy
                pick_env_idx = subtask_type == 0
                place_env_idx = subtask_type == 1
                navigate_env_idx = subtask_type == 2
                open_env_idx = subtask_type == 3
                close_env_idx = subtask_type == 4

                # get targ names to query per-obj policies
                sapien_obj_names = [None] * uenv.num_envs
                for env_num, subtask_num in enumerate(torch.clip(subtask_pointer, max=len(uenv.task_plan) - 1)):
                    subtask = uenv.task_plan[subtask_num]
                    if isinstance(subtask, PickSubtask) or isinstance(subtask, PlaceSubtask):
                        sapien_obj_names[env_num] = uenv.subtask_objs[subtask_num]._objs[env_num].name
                    elif isinstance(subtask, OpenSubtask) or isinstance(subtask, CloseSubtask):
                        sapien_obj_names[env_num] = uenv.subtask_articulations[subtask_num]._objs[env_num].name
                targ_names = []
                for sapien_on in sapien_obj_names:
                    if sapien_on is None:
                        targ_names.append(None)
                    else:
                        for tn in task_targ_names:
                            if tn in sapien_on:
                                targ_names.append(tn)
                                break
                assert len(targ_names) == uenv.num_envs

                # if policy_type == "rl_per_obj" or doing open/close env, need to query per-obj policy
                if POLICY_TYPE == "rl_per_obj" or torch.any(open_env_idx) or torch.any(close_env_idx):
                    tn_env_idxs = dict()
                    for env_num, tn in enumerate(targ_names):
                        if tn not in tn_env_idxs:
                            tn_env_idxs[tn] = []
                        tn_env_idxs[tn].append(env_num)
                    for k, v in tn_env_idxs.items():
                        bool_env_idx = torch.zeros(uenv.num_envs, dtype=torch.bool)
                        bool_env_idx[v] = True
                        tn_env_idxs[k] = bool_env_idx

                # query appropriate policy and place in action
                def set_subtask_targ_policy_act(subtask_name, subtask_env_idx):
                    if (
                        POLICY_TYPE == "rl_per_obj"
                        or subtask_name
                        in [
                            "open",
                            "close",
                        ]
                    ) and subtask_name != "navigate":
                        for tn, targ_env_idx in tn_env_idxs.items():
                            subtask_targ_env_idx = subtask_env_idx & targ_env_idx
                            if torch.any(subtask_targ_env_idx):
                                action[subtask_targ_env_idx] = policies[subtask_name][tn](
                                    recursive_slice(obs, subtask_targ_env_idx)
                                )
                    else:
                        action[subtask_env_idx] = policies[subtask_name]["all"](recursive_slice(obs, subtask_env_idx))

                if torch.any(pick_env_idx):
                    set_subtask_targ_policy_act("pick", pick_env_idx)
                if torch.any(place_env_idx):
                    set_subtask_targ_policy_act("place", place_env_idx)
                if torch.any(navigate_env_idx):
                    set_subtask_targ_policy_act("navigate", navigate_env_idx)
                if torch.any(open_env_idx):
                    set_subtask_targ_policy_act("open", open_env_idx)
                if torch.any(close_env_idx):
                    set_subtask_targ_policy_act("close", close_env_idx)

                return action

    # -------------------------------------------------------------------------------------------------
    # RUN
    # -------------------------------------------------------------------------------------------------

    task_targ_names = set()
    for subtask_name in POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS["rl"][task]:
        task_targ_names.update(POLICY_TYPE_TASK_SUBTASK_TO_TARG_IDS["rl"][task][subtask_name])

    eval_obs = to_tensor(eval_envs.reset(seed=SEED, options={})[0], device=device, dtype="float")
    subtask_fail_counts = defaultdict(int)
    last_subtask_pointer = uenv.subtask_pointer.clone()
    pbar = tqdm(range(MAX_TRAJECTORIES), total=MAX_TRAJECTORIES)
    step_num = 0

    # Print debug header once
    debug_header_printed = False
    debug_line_count = 0

    def check_done():
        if SAVE_TRAJECTORIES:
            # NOTE (arth): eval_envs.env._env is bad, fix in wrappers instead (prob with get_attr func)
            return eval_envs.env._env.reached_max_trajectories
        return len(eval_envs.return_queue) >= MAX_TRAJECTORIES

    def update_pbar(step_num):
        if SAVE_TRAJECTORIES:
            diff = eval_envs.env._env.num_saved_trajectories - pbar.last_print_n
        else:
            diff = len(eval_envs.return_queue) - pbar.last_print_n

        if diff > 0:
            pbar.update(diff)

        pbar.set_description(f"{step_num=}")

    def update_fail_subtask_counts(done):
        nonlocal debug_line_count
        if torch.any(done):
            # Get indices of done environments
            done_env_indices = torch.where(done)[0]
            total_subtasks = len(uenv.task_plan)

            # Process each done environment individually
            for env_idx in done_env_indices:
                fail_subtask = last_subtask_pointer[env_idx].item()
                subtask_fail_counts[fail_subtask] += 1

                success_status = (
                    "✓ SUCCESS"
                    if fail_subtask >= total_subtasks
                    else f"✗ FAILED at subtask {fail_subtask}/{total_subtasks}"
                )
                will_save = (
                    DEMO_FILTER == "any"
                    or (DEMO_FILTER == "min_success_subtasks" and fail_subtask >= MIN_SUCCESS_SUBTASKS)
                    or (DEMO_FILTER == "success" and fail_subtask >= total_subtasks)
                )
                save_status = "[WILL SAVE]" if will_save else "[FILTERED OUT]"
                # Print newline to preserve status, then print message
                print()
                print(f"📊 Env {env_idx} ended at step {step_num}: {success_status} {save_status}")

            # Reset counter so status prints fresh after messages
            debug_line_count = 0

            with open(logger.exp_path / "subtask_fail_counts.json", "w+") as f:
                json.dump(
                    dict((str(k), int(subtask_fail_counts[k])) for k in sorted(subtask_fail_counts.keys())),
                    f,
                )

    while not check_done():
        timer.end("other")
        last_subtask_pointer = uenv.subtask_pointer.clone()
        action = act(eval_obs)
        timer.end("sample")
        eval_obs, _, term, trunc, info = eval_envs.step(action)
        timer.end("sim_sample")

        ################################################################################################################
        ### DEBUG PRINTING
        ################################################################################################################

        # Debug: Print subtask progression and success criteria with dynamic updates
        curr_subtask = uenv.subtask_pointer[0].item()

        # Print header once
        if not debug_header_printed:
            print(f"\n{'=' * 80}")
            print("SUBTASK PROGRESS MONITOR")
            print(f"{'=' * 80}")
            debug_header_printed = True

        # Build the status lines
        status_lines = []
        status_lines.append(f"[Step {step_num:5d}] Subtask: {curr_subtask}/{len(uenv.task_plan)}")

        # is_success
        if "is_success" in info:
            is_succ = info["is_success"]
            is_succ_val = is_succ[0] if hasattr(is_succ, "__len__") else is_succ
            status_lines.append(f"  is_success: {is_succ_val}")

        # Navigation criteria
        nav_keys = [
            "navigated_close",
            "oriented_correctly",
            "cumulative_force_within_limit",
            "robot_rest",
            "ee_rest",
            "is_static",
            "is_grasped",
        ]
        nav_status = []
        for key in nav_keys:
            if key in info:
                val = info[key]
                val_display = val[0] if hasattr(val, "__len__") and len(val) > 0 else val
                # Use checkmark/cross for boolean values
                if isinstance(val_display, (bool, np.bool_)):
                    symbol = "✓" if val_display else "✗"
                    nav_status.append(f"{key}:{symbol}")
                else:
                    nav_status.append(f"{key}:{val_display}")
        if nav_status:
            for nav_item in nav_status:
                status_lines.append(f"  Nav {nav_item}")

        # Show individual joint deviations for robot_rest
        if "robot_rest" in info and "is_grasped" in info:
            is_grasped_val = info["is_grasped"][0] if hasattr(info["is_grasped"], "__len__") else info["is_grasped"]
            # Compute robot_rest_dist like in sequential_task.py
            joint_names = [
                "torso",
                "head_pan",
                "shoulder_pan",
                "head_tilt",
                "shoulder_lift",
                "upperarm_roll",
                "elbow_flex",
                "forearm_roll",
                "wrist_flex",
                "wrist_roll",
            ]
            if is_grasped_val:
                # When grasping, check joints 3:-2 (torso + arm)
                robot_rest_dist = torch.abs(uenv.agent.robot.qpos[0, 3:-2] - uenv.resting_qpos)
                tolerance = uenv.navigate_cfg.robot_resting_qpos_tolerance_grasping
                tolerances = [tolerance] * len(robot_rest_dist)
            else:
                # When not grasping, check torso (joint 3) separately + arm (4:-2)
                # Torso has stricter tolerance of 0.01 vs standard tolerance
                torso_dist = torch.abs(uenv.agent.robot.qpos[0, 3] - uenv.resting_qpos[0])
                arm_rest_dist = torch.abs(uenv.agent.robot.qpos[0, 4:-2] - uenv.resting_qpos[1:])
                # Combine torso + arm distances
                robot_rest_dist = torch.cat([torso_dist.unsqueeze(0), arm_rest_dist])
                tolerance = uenv.navigate_cfg.robot_resting_qpos_tolerance
                # Torso has stricter tolerance of 0.01
                tolerances = [0.01] + [tolerance] * len(arm_rest_dist)

            # Display each joint's deviation
            status_lines.append("  Joint deviations:")
            joint_within = []
            for name, dist, tol in zip(joint_names, robot_rest_dist, tolerances):
                within_limit = dist < tol
                joint_within.append(within_limit)
                symbol = "✓" if within_limit else "✗"
                status_lines.append(f"    {name}: {dist.item():.4f} (tol={tol:.3f}) {symbol}")
            # Extra debug info
            status_lines.append(f"resting_qpos: {uenv.resting_qpos[1:]}")
            status_lines.append(f"uenv.agent.robot.qpos[0, 4:-2]: {uenv.agent.robot.qpos[0, 4:-2]}")
            status_lines.append(f"uenv.resting_qpos[0]: {uenv.resting_qpos[0]}")
            status_lines.append(f"uenv.agent.robot.qpos[0, 3]: {uenv.agent.robot.qpos[0, 3]}")

        # Other info
        if "subtasks_steps_left" in info:
            val = info["subtasks_steps_left"]
            val_display = val[0] if hasattr(val, "__len__") and len(val) > 0 else val
            status_lines.append(f"  Steps left: {val_display}")

        # Move cursor up if we've printed before, otherwise just print
        if debug_line_count > 0:
            # Move cursor up and clear each line
            sys.stdout.write(f"\033[{debug_line_count}A")
            for _ in range(debug_line_count):
                sys.stdout.write("\033[K")  # Clear line
                sys.stdout.write("\n")
            sys.stdout.write(f"\033[{debug_line_count}A")

        # Print all status lines
        for line in status_lines:
            print(line)

        debug_line_count = len(status_lines)
        sys.stdout.flush()

        ################################################################################################################

        eval_obs = to_tensor(
            eval_obs,
            device=device,
            dtype="float",
        )
        update_pbar(step_num)
        update_fail_subtask_counts(term | trunc)
        if POLICY_KEY == "dp":
            if torch.any(term | trunc):
                dp_action_history.clear()
        step_num += 1

    # -------------------------------------------------------------------------------------------------
    # PRINT/SAVE RESULTS
    # -------------------------------------------------------------------------------------------------

    print(
        "subtask_fail_counts",
        dict((k, subtask_fail_counts[k]) for k in sorted(subtask_fail_counts.keys())),
    )

    results_logs = dict(
        num_trajs=len(eval_envs.return_queue),
        return_per_step=common.to_tensor(eval_envs.return_queue, device=device).float().mean()
        / eval_envs.max_episode_steps,
        success_once=common.to_tensor(eval_envs.success_once_queue, device=device).float().mean(),
        success_at_end=common.to_tensor(eval_envs.success_at_end_queue, device=device).float().mean(),
        len=common.to_tensor(eval_envs.length_queue, device=device).float().mean(),
    )
    time_logs = timer.get_time_logs(pbar.last_print_n * eval_envs.max_episode_steps)
    print(
        "results",
        results_logs,
    )
    print("time", time_logs)
    print("total_time", timer.total_time_elapsed)

    with open(logger.exp_path / "output.txt", "w") as f:
        f.write("results\n" + str(results_logs) + "\n")
        f.write("time\n" + str(time_logs) + "\n")

    # -------------------------------------------------------------------------------------------------
    # CLOSE
    # -------------------------------------------------------------------------------------------------

    eval_envs.close()
    logger.close()


if __name__ == "__main__":
    import sys

    eval(task=sys.argv[1], task_plan_path=sys.argv[2])
