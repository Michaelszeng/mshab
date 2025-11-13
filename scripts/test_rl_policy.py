import types
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import torch
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from mani_skill import ASSET_DIR
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mshab.envs
from mshab.agents.ppo import Agent as PPOAgent
from mshab.agents.sac import Agent as SACAgent
from mshab.envs.planner import plan_data_from_file
from mshab.envs.wrappers.action import FetchActionWrapper
from mshab.envs.wrappers.observation import FetchDepthObservationWrapper, FrameStack
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg

human_conf = dict(
    num_envs=1,
    render_mode="human",
    shader_dir="minimal",
    require_build_configs_repeated_equally_across_envs=False,
)
default_conf = dict(
    num_envs=252,  # RCAD has 63 train scenes, so 252 envs -> 4 parallel envs reserved for each scene
    render_mode="rgb_array",
    shader_dir="minimal",
)

CONF = human_conf

task = "set_table"  # "tidy_house", "prepare_groceries", or "set_table"
subtask = (
    "pick"  # "pick", "place", "open", "close", "navigate" (NOT "sequential" - no RL policy for full sequential task)
)
# NOTE: This script tests individual subtask RL policies
#     To test the full sequential task, you would need to load and switch between multiple policies
split = "train"  # "train", "val"
target_id = "all"  # e.g. "024_bowl", "013_apple", or "all"

# Validate subtask selection
valid_subtasks = ["pick", "place", "open", "close", "navigate"]
if subtask not in valid_subtasks:
    raise ValueError(
        f"subtask must be one of {valid_subtasks} for RL policy testing. "
        f"Got '{subtask}'. Note: 'sequential' is not supported as RL policies "
        f"are only trained on individual subtasks."
    )

# Load task plans - always use "all.json"
# The SubtaskTrain environment will automatically filter episodes based on spawn_data
REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
plan_data = plan_data_from_file(REARRANGE_DIR / "task_plans" / task / subtask / split / "all.json")
spawn_data_fp = REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"

extra_gym_kwargs = dict(
    human_render_camera_configs=dict(
        width=640,
        height=480,
    ),
)

# Create the SubtaskTrain environment
env_id = f"{subtask.capitalize()}SubtaskTrain-v0"
env = gym.make(
    env_id,
    **CONF,
    obs_mode="depth",  # RL policies were trained with depth observations
    sim_backend="cpu",
    robot_uids="fetch",
    control_mode="pd_joint_delta_pos",
    reward_mode="normalized_dense",
    max_episode_steps=200,  # Longer for testing; training used 100
    task_plans=plan_data.plans,
    scene_builder_cls=plan_data.dataset,
    spawn_data_fp=spawn_data_fp,
    **extra_gym_kwargs,
)

# Apply observation wrappers to match the structure used in training
# This converts obs from ['agent', 'extra', 'sensor_param', 'sensor_data']
# to ['state', 'pixels'] which is what RL policies expect
env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
env = FrameStack(
    env,
    num_stack=3,  # frame_stack used during training
    stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
)

# Apply action wrapper to constrain robot joints - MUST match training!
# This zeroes out action dimensions for joints that should be stationary
env = FetchActionWrapper(
    env,
    stationary_base=False,  # Base can move (trained with moving base)
    stationary_torso=False,  # Torso can move (trained with moving torso)
    stationary_head=True,  # Head is fixed (trained with fixed head)
)

# LOAD RL POLICY
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_dir = (
    Path(ASSET_DIR) / "mshab_checkpoints" / "rl" / task / subtask / target_id
    if (Path(ASSET_DIR) / "mshab_checkpoints").exists()
    else Path("mshab_checkpoints") / "rl" / task / subtask / target_id
)
cfg_path = ckpt_dir / "config.yml"
ckpt_path = ckpt_dir / "policy.pt"

# Parse config to detect algorithm type
algo_cfg = parse_cfg(default_cfg_path=cfg_path).algo
print(f"Config path: {cfg_path}")
print(f"Checkpoint path: {ckpt_path}")
print(f"Detected algorithm: {algo_cfg.name.upper()}")
print(f"Environment observation space keys: {list(env.observation_space.spaces.keys())}")


if algo_cfg.name == "sac":
    # SAC needs pixel and state observation spaces
    obs_space = env.observation_space

    # The observation space has "pixels" and "state" keys after being wrapped
    if "pixels" in obs_space.spaces:
        pixels_obs_space: spaces.Dict = obs_space["pixels"]
        state_obs_space: spaces.Box = obs_space["state"]
    else:
        raise ValueError(
            "SAC expects observation space with 'pixels' and 'state' keys. "
            "Your environment has keys: "
            + str(list(obs_space.spaces.keys()))
            + ". You may need to wrap the environment with appropriate wrappers."
        )

    # Reshape pixel observation space for frame-stacked images
    # Vectorized env obs are 5D: (batch, frame_stack, depth, height, width)
    # SAC expects 3D: (frame_stack * depth, height, width)
    model_pixel_obs_space = dict()
    for k, space in pixels_obs_space.items():
        shape, low, high, dtype = (
            space.shape,
            space.low,
            space.high,
            space.dtype,
        )
        print(f"  Pixel obs '{k}' original shape: {shape}")
        if len(shape) == 5:  # (batch, frame_stack, depth, height, width)
            # Remove batch dim and flatten frame_stack * depth
            shape = (shape[1] * shape[2], shape[-2], shape[-1])
            low = low[0].reshape((-1, *low.shape[-2:]))
            high = high[0].reshape((-1, *high.shape[-2:]))
            print(f"  Pixel obs '{k}' reshaped to: {shape}")
        elif len(shape) == 4:  # (frame_stack, depth, height, width) - no batch dim
            shape = (shape[0] * shape[1], shape[-2], shape[-1])
            low = low.reshape((-1, *low.shape[-2:]))
            high = high.reshape((-1, *high.shape[-2:]))
            print(f"  Pixel obs '{k}' reshaped to: {shape}")
        model_pixel_obs_space[k] = spaces.Box(low, high, shape, dtype)
    model_pixel_obs_space = spaces.Dict(model_pixel_obs_space)
    print(f"Model pixel obs space keys: {list(model_pixel_obs_space.spaces.keys())}")

    # Create SAC agent with all required hyperparameters
    policy = SACAgent(
        model_pixel_obs_space,
        state_obs_space.shape,
        env.action_space.shape,
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
    policy.load_state_dict(torch.load(ckpt_path, map_location=device)["agent"])
    policy.to(device)

    def policy_act(obs_np):
        """SAC: takes separate pixels and state inputs"""
        with torch.no_grad():
            obs_tensor = to_tensor(obs_np, device=device, dtype="float")
            act_tensor = policy.actor(
                obs_tensor["pixels"],
                obs_tensor["state"],
                compute_pi=False,
                compute_log_pi=False,
            )[0]
            return act_tensor.cpu().numpy()

else:
    raise NotImplementedError(f"Algorithm {algo_cfg.name} not supported")


# Run simulation loop
obs, info = env.reset()
while True:
    env.render()  # comment out if you spawned many envs headless
    action = policy_act(obs)
    # action = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(action)
    if term or trunc:
        obs, info = env.reset()

# add env wrappers here
if CONF != human_conf:
    env = RecordVideo(env, "videos/", episode_trigger=lambda i: True, disable_logger=True)

    env = ManiSkillVectorEnv(
        env,
        max_episode_steps=1000,  # set manually based on task
        ignore_terminations=True,  # set to False for partial resets
    )

    # add vector env wrappers here
