from ajet.utils.metric_helper.save_trajectory_as_json import save_trajectory_as_json
from ajet.utils.metric_helper.tool_metric_helper import compute_tool_metrics_from_trajectories
from ajet.utils.metric_helper.reward_metric_helper import compute_reward_metrics_from_trajectories


def save_trajectory_as_json_file(ctx_trackers, global_steps, config, prefix):
    if config.ajet.trainer_common.save_trajectory_as_json_file:
        save_trajectory_as_json(ctx_trackers, global_steps, prefix)

def update_metrics(context_tracker_arr, metrics:dict, prefix):
    tool_metrics = compute_tool_metrics_from_trajectories(context_tracker_arr, prefix)
    reward_metrics = compute_reward_metrics_from_trajectories(context_tracker_arr, prefix)
    if tool_metrics:
        metrics.update(tool_metrics)
    if reward_metrics:
        metrics.update(reward_metrics)
    return
