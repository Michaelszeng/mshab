import json
from pathlib import Path


def shorten_task_plans(input_path, output_path):
    """
    Creates a shortened version of task plans containing only the first:
    nav -> open -> nav -> pick -> nav -> place subtasks
    """
    # Read the original JSON
    with open(input_path, "r") as f:
        data = json.load(f)

    # Create new shortened plans
    shortened_plans = []

    for plan in data["plans"]:
        subtasks = plan["subtasks"]

        # Track which subtasks we've found
        target_sequence = ["navigate", "open", "navigate", "pick", "navigate", "place"]
        shortened_subtasks = []
        target_idx = 0

        # Iterate through subtasks and collect the ones we want
        for subtask in subtasks:
            if target_idx < len(target_sequence):
                if subtask["type"] == target_sequence[target_idx]:
                    shortened_subtasks.append(subtask)
                    target_idx += 1

        # Only include plans that have the complete sequence
        if len(shortened_subtasks) == len(target_sequence):
            shortened_plan = {
                "build_config_name": plan["build_config_name"],
                "init_config_name": plan["init_config_name"],
                "subtasks": shortened_subtasks,
            }
            shortened_plans.append(shortened_plan)
        else:
            print(
                f"Warning: Plan with build_config '{plan['build_config_name']}' "
                f"doesn't have complete sequence (found {len(shortened_subtasks)}/{len(target_sequence)})"
            )

    # Create output data structure
    output_data = {"dataset": data["dataset"], "plans": shortened_plans}

    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Created shortened task plans with {len(shortened_plans)} plans")
    print(f"Original had {len(data['plans'])} plans")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    input_path = Path(
        "/home/michzeng/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/sequential/train/all.json"
    )
    output_path = Path(
        "/home/michzeng/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/set_table/sequential/train/all_short.json"
    )

    shorten_task_plans(input_path, output_path)
