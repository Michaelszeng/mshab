"""
Janky Script to create easier task plans (for which the RL policies are more likely to succeed and will be easier for a
diffusion policy to learn the entire sequence).

In particular, we just take a chunk of the original task plan.

If REMOVE_FIRST_NAVIGATE is False, then we take the first pick + place sequence.
If REMOVE_FIRST_NAVIGATE is True, then we take the first pick + place sequence but remove the first navigate subtask.
"""

import json
from pathlib import Path

import torch

TASK = "tidy_house"  # "set_table" or "tidy_house" or "prepare_groceries"
REMOVE_FIRST_NAVIGATE = True  # If True, removes first "navigate" and updates spawn locations

SPLIT = "train"

if TASK == "set_table":
    TARGET_SEQUENCE = ["navigate", "open", "navigate", "pick", "navigate", "place"]
elif TASK == "tidy_house":
    TARGET_SEQUENCE = ["navigate", "pick", "navigate", "place"]
elif TASK == "prepare_groceries":
    TARGET_SEQUENCE = ["navigate", "pick", "navigate", "place"]
else:
    raise ValueError(f"Invalid task: {TASK}")

# Remove first navigate if requested
if REMOVE_FIRST_NAVIGATE:
    if TARGET_SEQUENCE[0] == "navigate":
        TARGET_SEQUENCE = TARGET_SEQUENCE[1:]
        print(f"Removed first 'navigate'. New sequence: {TARGET_SEQUENCE}")
    else:
        raise ValueError("REMOVE_FIRST_NAVIGATE is True but first element is not 'navigate'")


def generate_new_spawn_data(shortened_plans):
    """Create a minimal spawn_data dict that teleports the robot for the *first* subtask
    of each shortened sequential task plan.

    We reuse poses from the corresponding single-subtask dataset (open or pick).
    """
    subtask_type = TARGET_SEQUENCE[0]  # "open" or "pick"

    single_src = (
        Path.home()
        / f".maniskill/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/{TASK}/{subtask_type}/{SPLIT}/spawn_data.pt"
    )
    assert single_src.exists(), f"Could not find {single_src}"
    print("loading single-subtask poses:", single_src)
    single_sd = torch.load(single_src, map_location="cpu")

    # pick an arbitrary sample pose (or the first) – could be scene-matched but usually fine
    sample_pose = next(iter(single_sd.values()))

    new_sd = {}
    for plan in shortened_plans:
        uid = plan["subtasks"][0]["uid"]
        entry = {}
        for k in ("robot_pos", "robot_qpos"):
            if k in sample_pose:
                entry[k] = sample_pose[k].clone()
        new_sd[uid] = entry

    dst = (
        Path.home()
        / f".maniskill/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/{TASK}/sequential/{SPLIT}/spawn_data_short_remove_first_navigate.pt"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_sd, dst)
    print("Saved new spawn_data to", dst)
    return dst


def shorten_task_plans(input_path, output_path):
    """
    Creates a shortened version of task plans containing only the first:
    nav -> open -> nav -> pick -> nav -> place subtasks

    If REMOVE_FIRST_NAVIGATE is True, also updates spawn locations to match
    the new first subtask.
    """
    # Read the original JSON
    with open(input_path, "r") as f:
        data = json.load(f)

    # Create new shortened plans
    shortened_plans = []

    for plan in data["plans"]:
        subtasks = plan["subtasks"]

        # Track which subtasks we've found
        shortened_subtasks = []
        target_idx = 0

        # Iterate through subtasks and collect the ones we want
        for subtask in subtasks:
            if target_idx < len(TARGET_SEQUENCE):
                if subtask["type"] == TARGET_SEQUENCE[target_idx]:
                    shortened_subtasks.append(subtask)
                    target_idx += 1

        # Only include plans that have the complete sequence
        if len(shortened_subtasks) == len(TARGET_SEQUENCE):
            shortened_plan = {
                "build_config_name": plan["build_config_name"],
                "init_config_name": plan["init_config_name"],
                "subtasks": shortened_subtasks,
            }
            shortened_plans.append(shortened_plan)

            # collect for spawn data later
            pass
        else:
            print(
                f"Warning: Plan with build_config '{plan['build_config_name']}' "
                f"doesn't have complete sequence (found {len(shortened_subtasks)}/{len(TARGET_SEQUENCE)})"
            )

    # Create output data structure
    output_data = {"dataset": data["dataset"], "plans": shortened_plans}

    if REMOVE_FIRST_NAVIGATE:
        generate_new_spawn_data(shortened_plans)

    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Created shortened task plans with {len(shortened_plans)} plans")
    print(f"Original had {len(data['plans'])} plans")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    input_path = Path(
        f"/home/michzeng/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/{TASK}/sequential/train/all.json"
    )

    if REMOVE_FIRST_NAVIGATE:
        output_path = Path(
            f"/home/michzeng/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/{TASK}/sequential/train/all_short_remove_first_navigate.json"
        )
    else:
        output_path = Path(
            f"/home/michzeng/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/{TASK}/sequential/train/all_short.json"
        )

    shorten_task_plans(input_path, output_path)
