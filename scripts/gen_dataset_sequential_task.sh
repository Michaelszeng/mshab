#!/usr/bin/bash

if [[ -z "${MS_ASSET_DIR}" ]]; then
    MS_ASSET_DIR="$HOME/.maniskill"
fi

TASKS=(
    # tidy_house
    # prepare_groceries
    set_table
)

# TASK_PLAN_PATH="scene_datasets/replica_cad_dataset/rearrange/task_plans/{task}/sequential/train/all_short_remove_first_navigate.json"
TASK_PLAN_PATH="scene_datasets/replica_cad_dataset/rearrange/task_plans/{task}/sequential/train/all_short.json"

for task in "${TASKS[@]}"
do
    python -m mshab.utils.gen.gen_data_sequential_task "$task" "$TASK_PLAN_PATH"
done