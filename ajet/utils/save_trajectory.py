import os
import json
from ajet.utils.msg_converter import convert_grouped_steps_to_openai_format


def save_train_trajectory(ctx_trackers, global_steps):
    """Save training ctx_trackers to JSON files."""
    for ctx_tracker in ctx_trackers:
        reward = ctx_tracker.reward_structure.raw_reward
        if reward >= 1:
            ctx_tracker.tag = "success"
        elif reward == 0:
            ctx_tracker.tag = "failure"
        else:
            ctx_tracker.tag = "half_success"
        
        # Use unified conversion function to convert grouped_steps to OpenAI format
        if hasattr(ctx_tracker, 'get_grouped_steps_openai_format'):
            formatted_traj = ctx_tracker.get_grouped_steps_openai_format()
        else:
            formatted_traj = convert_grouped_steps_to_openai_format(ctx_tracker.grouped_steps)

        traj_data = {
            "task_id": ctx_tracker.task_id,
            "task_tag": ctx_tracker.tag,
            "reward_structure": ctx_tracker.reward_structure.model_dump(),
            "traj": formatted_traj
        }
        # Extract reward_stats from workflow_metadata
        if hasattr(ctx_tracker, 'workflow_metadata') and ctx_tracker.workflow_metadata:
            if 'reward_stats' in ctx_tracker.workflow_metadata:
                traj_data['reward_structure']['reward_stats'] = ctx_tracker.workflow_metadata['reward_stats']
        
        traj_save_dir = os.path.join(
            os.environ.get("BEST_LOGGER_PATH", "launcher_record"),
            "ctx_trackers",
            "train",
            f"step_{global_steps}"
        )
        os.makedirs(traj_save_dir, exist_ok=True)
        traj_file_path = os.path.join(traj_save_dir, f"{ctx_tracker.task_id}.json")
        
        with open(traj_file_path, "w", encoding="utf-8") as f:
            json.dump(traj_data, f, ensure_ascii=False, indent=2)


def save_eval_trajectory(ctx_trackers, global_steps):
    """Save evaluation ctx_trackers to JSON files."""
    for ctx_tracker in ctx_trackers:
        # Use unified conversion function to convert grouped_steps to OpenAI format
        if hasattr(ctx_tracker, 'get_grouped_steps_openai_format'):
            formatted_traj = ctx_tracker.get_grouped_steps_openai_format()
        else:
            formatted_traj = convert_grouped_steps_to_openai_format(ctx_tracker.grouped_steps)

        traj_data = {
            "task_id": ctx_tracker.task_id,
            "task_tag": ctx_tracker.tag,
            "reward_structure": ctx_tracker.reward_structure.model_dump(),
            "traj": formatted_traj
        }
        
        # Extract reward_stats from workflow_metadata
        if hasattr(ctx_tracker, 'workflow_metadata') and ctx_tracker.workflow_metadata:
            if 'reward_stats' in ctx_tracker.workflow_metadata:
                traj_data['reward_structure']['reward_stats'] = ctx_tracker.workflow_metadata['reward_stats']
        
        traj_save_dir = os.path.join(
            os.environ.get("BEST_LOGGER_PATH", "launcher_record"),
            "ctx_trackers",
            "val",
            f"step_{global_steps}"
        )
        os.makedirs(traj_save_dir, exist_ok=True)
        traj_file_path = os.path.join(traj_save_dir, f"{ctx_tracker.task_id}.json")
        
        with open(traj_file_path, "w", encoding="utf-8") as f:
            json.dump(traj_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved trajectory to {traj_file_path}")