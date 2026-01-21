import os
import json
from ajet.utils.msg_converter import convert_grouped_steps_to_openai_format


def save_trajectory_as_json(ctx_trackers, global_steps, prefix="train"):
    """
    Save ctx_trackers to JSON files for either training or evaluation.

    Args:
        ctx_trackers (list): List of context trackers containing trajectory data.
        global_steps (int): The global step count to organize saved files.
        prefix (str): Directory prefix indicating the type of trajectory ("train" or "eval").
    """
    for ctx_tracker in ctx_trackers:
        # Determine task tag based on reward
        reward = ctx_tracker.reward_structure.raw_reward
        if reward >= 1:
            ctx_tracker.tag = "success"
        elif reward == 0:
            ctx_tracker.tag = "failure"
        else:
            ctx_tracker.tag = "half_success"

        formatted_traj = convert_grouped_steps_to_openai_format(ctx_tracker.saved_timelines)

        # Prepare trajectory data
        traj_data = {
            "task_id": ctx_tracker.task_id,
            "task_tag": ctx_tracker.tag,
            "reward_structure": ctx_tracker.reward_structure.model_dump(),
            "traj": formatted_traj
        }

        # Extract reward_stats from workflow_metadata if available
        if hasattr(ctx_tracker, 'workflow_metadata') and ctx_tracker.workflow_metadata:
            if 'reward_stats' in ctx_tracker.workflow_metadata:
                traj_data['reward_structure']['reward_stats'] = ctx_tracker.workflow_metadata['reward_stats']

        # Define save directory and file path
        traj_save_dir = os.path.join(
            os.environ.get("BEST_LOGGER_PATH", "launcher_record"),
            "ctx_trackers",
            prefix,
            f"step_{global_steps}"
        )
        os.makedirs(traj_save_dir, exist_ok=True)
        traj_file_path = os.path.join(traj_save_dir, f"{ctx_tracker.task_id}.json")

        # Save trajectory data to JSON file
        with open(traj_file_path, "w", encoding="utf-8") as f:
            json.dump(traj_data, f, ensure_ascii=False, indent=2)


        print(f"Saved trajectory to {traj_file_path}")
