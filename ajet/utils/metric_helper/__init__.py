from ajet.utils.metric_helper.save_trajectory_as_json import save_trajectory_as_json
from ajet.utils.metric_helper.tool_metric_helper import compute_tool_metrics_from_trajectories
from ajet.utils.metric_helper.reward_metric_helper import compute_reward_metrics_from_trajectories


def save_trajectory_as_json_file(ctx_trackers, global_steps, config, prefix):
    if config.ajet.trainer_common.save_trajectory_as_json_file:
        save_trajectory_as_json(ctx_trackers, global_steps, prefix)

def update_metrics(context_tracker_arr, metrics:dict, prefix):
    # Debug: Check log_metrics content
    print(f"[update_metrics] called with prefix={prefix}, num_trackers={len(context_tracker_arr)}")
    for i, traj in enumerate(context_tracker_arr[:3]):  # Check first 3
        has_log_metrics = hasattr(traj, 'log_metrics') and traj.log_metrics
        print(f"[update_metrics] traj[{i}] has log_metrics: {has_log_metrics}")
        if has_log_metrics:
            print(f"[update_metrics] traj[{i}].log_metrics keys: {list(traj.log_metrics.keys())}")
    
    tool_metrics = compute_tool_metrics_from_trajectories(context_tracker_arr, prefix)
    reward_metrics = compute_reward_metrics_from_trajectories(context_tracker_arr, prefix)
    
    print(f"[update_metrics] tool_metrics count: {len(tool_metrics)}, reward_metrics count: {len(reward_metrics)}")
    
    if tool_metrics:
        metrics.update(tool_metrics)
    if reward_metrics:
        metrics.update(reward_metrics)
    return
