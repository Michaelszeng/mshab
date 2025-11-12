"""
Test script for visualizing dataset episodes.

This script loads recorded demonstration data from HDF5 files and replays
the actions in the environment. Unlike test_rl_policy.py which uses a trained
policy to generate actions, this script directly uses the actions stored in
the dataset.

Usage:
    1. Configure task, subtask, split, and dataset paths below
    2. Set episode_idx to select which episode to visualize
    3. Set object_file_idx to select which object file to load
    4. Optionally enable loop_episodes to cycle through all episodes
    5. Run the script to see the visualization

The environment setup matches the training configuration, including the same
observation wrappers and action wrappers used during data collection.
"""

import json
import os
import time
from pathlib import Path

# Expand MS_ASSET_DIR BEFORE importing ManiSkill
# The scene builder constructs paths using ASSET_DIR and they must not contain ~
if "MS_ASSET_DIR" in os.environ and os.environ["MS_ASSET_DIR"].startswith("~"):
    os.environ["MS_ASSET_DIR"] = str(Path(os.environ["MS_ASSET_DIR"]).expanduser())

import gymnasium as gym
import h5py
import mani_skill.envs
import numpy as np
import torch
from mani_skill import ASSET_DIR

import mshab.envs
from mshab.envs.planner import plan_data_from_file
from mshab.envs.wrappers.action import FetchActionWrapper
from mshab.envs.wrappers.observation import FetchDepthObservationWrapper, FrameStack
from mshab.train_diffusion_policy import DPDataset

# Configuration
task = "set_table"  # "tidy_house", "prepare_groceries", or "set_table"
subtask = "pick"  # "sequential", "pick", "place", "open", "close"
split = "train"  # "train", "val"
target_id = "all"  # e.g. "024_bowl", "013_apple", or "all"

# Dataset configuration - DPDataset mode
use_dp_dataset = True  # If True, use DPDataset with observation/action horizons
obs_horizon = 2  # Observation horizon for diffusion policy
pred_horizon = 16  # Prediction horizon for diffusion policy
act_horizon = 8  # Action horizon (how many actions to execute)
trajs_per_obj = "all"  # Number of trajectories to load per object file
sample_idx = 0  # Which sample from the dataset to visualize (dataset provides pre-computed slices)
loop_samples = False  # If True, continuously loop through dataset samples
show_action_details = False  # If True, print action values each step

# Load task plans
REARRANGE_DIR = (Path(ASSET_DIR) / "scene_datasets/replica_cad_dataset/rearrange").expanduser()
plan_data = plan_data_from_file(REARRANGE_DIR / "task_plans" / task / subtask / split / "all.json")
spawn_data_fp = REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"

# Find dataset directory
data_path = (Path(ASSET_DIR) / "scene_datasets/replica_cad_dataset/rearrange-dataset" / task / subtask).expanduser()

if not data_path.exists():
    print(f"ERROR: Dataset directory not found at {data_path.absolute()}")
    print("\nTo download the dataset, run:")
    dataset_dir = "$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange-dataset"
    print(f'  export MSHAB_DATASET_DIR="{dataset_dir}"')
    print(
        "  huggingface-cli download --repo-type dataset arth-shukla/MS-HAB-SetTable "
        '--local-dir "$MSHAB_DATASET_DIR/set_table"'
    )
    exit(1)

print(f"Loading dataset from: {data_path.absolute()}")

if use_dp_dataset:
    # Load dataset using DPDataset class (for diffusion policy visualization)
    print(f"\nLoading DPDataset with obs_horizon={obs_horizon}, pred_horizon={pred_horizon}...")
    dataset = DPDataset(
        data_path=data_path,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        control_mode="pd_joint_delta_pos",
        trajs_per_obj=trajs_per_obj,
        max_image_cache_size="all",  # Cache all images for visualization
        truncate_trajectories_at_success=False,
    )

    print(f"Dataset loaded! Total samples: {len(dataset)}")
    print("  - Each sample contains:")
    print(f"    - Observation sequence: {obs_horizon} steps")
    print(f"    - Action sequence: {pred_horizon} steps")
    print(f"  - We'll execute {act_horizon} actions per sample during visualization")

    if sample_idx >= len(dataset):
        print(f"\nERROR: sample_idx={sample_idx} is out of range (max={len(dataset) - 1})")
        exit(1)

    # Get first sample to show structure
    sample = dataset[sample_idx]
    print(f"\nSample {sample_idx} structure:")
    print(f"  - observations['state']: {sample['observations']['state'].shape}")
    if "fetch_head_depth" in sample["observations"]:
        print(f"  - observations['fetch_head_depth']: {sample['observations']['fetch_head_depth'].shape}")
    if "fetch_hand_depth" in sample["observations"]:
        print(f"  - observations['fetch_hand_depth']: {sample['observations']['fetch_hand_depth'].shape}")
    print(f"  - actions: {sample['actions'].shape}")

    actions = sample["actions"][:act_horizon].numpy()  # Extract act_horizon actions
    print(f"\nWill execute {len(actions)} actions from this sample")
else:
    # Original simple H5 loading (for raw episode replay)
    print("\nERROR: Non-DPDataset mode not implemented in this version.")
    print("Please set use_dp_dataset=True")
    exit(1)

# Create environment for visualization
human_conf = dict(
    num_envs=1,
    render_mode="human",
    shader_dir="minimal",
    require_build_configs_repeated_equally_across_envs=False,
)

extra_kwargs = dict(
    human_render_camera_configs=dict(
        width=640,
        height=480,
    ),
)

env_id = f"{subtask.capitalize()}SubtaskTrain-v0"
env = gym.make(
    env_id,
    # Simulation args
    **human_conf,
    obs_mode="depth",
    sim_backend="cpu",
    robot_uids="fetch",
    control_mode="pd_joint_delta_pos",
    # Rendering args
    reward_mode="normalized_dense",
    # TimeLimit args
    max_episode_steps=200,
    # SequentialTask args
    task_plans=plan_data.plans,
    scene_builder_cls=plan_data.dataset,
    # SubtaskTrain args
    spawn_data_fp=spawn_data_fp,
    # additional env_kwargs
    **extra_kwargs,
)

# Apply observation wrappers to match the training setup
env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
env = FrameStack(
    env,
    num_stack=3,
    stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
)

# Apply action wrapper to constrain robot joints
env = FetchActionWrapper(
    env,
    stationary_base=False,
    stationary_torso=False,
    stationary_head=True,
)

print("\nStarting visualization...")
print("The environment will replay actions from the DPDataset.")
print(f"Each sample executes {act_horizon} actions from a {pred_horizon}-action prediction.")
if loop_samples:
    print("Loop mode enabled - will cycle through dataset samples.")
print("\nPress Ctrl+C to exit.\n")

# Main loop - loop through dataset samples
current_sample_idx = sample_idx
try:
    while True:
        # Load current sample
        if current_sample_idx != sample_idx:
            sample = dataset[current_sample_idx]
            actions = sample["actions"][:act_horizon].numpy()
            print(f"\n{'=' * 60}")
            print(f"Sample {current_sample_idx}/{len(dataset)}")
            print(f"  - Executing {len(actions)} actions")
            print(f"{'=' * 60}")

        # Reset environment
        obs, info = env.reset()

        # Replay actions from the sample
        step = 0
        total_reward = 0.0

        while step < len(actions):
            env.render()

            # Get action from dataset sample
            action = actions[step]

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            # Display step info
            status = f"Step {step}/{len(actions)}: reward={reward:.4f}, total={total_reward:.4f}"
            if show_action_details:
                action_str = np.array2string(action, precision=3, suppress_small=True, max_line_width=100)
                print(f"{status}")
                print(f"  Action: {action_str}")
            else:
                print(status, end="\r")

            if terminated or truncated:
                print(f"\nEpisode ended at step {step}")
                print(f"Terminated: {terminated}, Truncated: {truncated}")
                print(f"Total reward: {total_reward:.4f}")
                break

        if step >= len(actions):
            print(f"\nCompleted all {len(actions)} actions")
            print(f"Total reward: {total_reward:.4f}")

        # If not looping, break after first sample
        if not loop_samples:
            break

        # Move to next sample (loop back to start if at end)
        current_sample_idx = (current_sample_idx + 1) % len(dataset)

        # Small delay before next sample
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\nInterrupted by user. Exiting...")

# Cleanup
print("Cleaning up...")
dataset.close()
env.close()
print("Done!")
