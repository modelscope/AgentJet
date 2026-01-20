"""
DeepFinance Tool Metrics Helper

Specialized module for extracting tool-related statistics and formatting SwanLab reports.
Extracts data from log_metrics['tool_stats'].

SwanLab metrics directory structure:
- tool_stats/           Overall statistics (success rate, cache hit rate, etc.)
- tool_time/            Time consumption statistics by tool
- tool_cache/           Cache hit rate by tool
- tool_error/           Error rate by tool
"""

from typing import List, Dict, Any
import numpy as np


def extract_tool_stats_from_trajectories(trajectories: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract tool_stats from trajectories list.

    Args:
        trajectories: List of trajectory objects containing log_metrics

    Returns:
        List of tool_stats dictionaries
    """
    tool_stats_list = []
    for traj in trajectories:
        if hasattr(traj, 'log_metrics') and traj.log_metrics:
            if 'tool_stats' in traj.log_metrics:
                tool_stats_list.append(traj.log_metrics['tool_stats'])
    return tool_stats_list



def compute_tool_metrics(tool_stats_list: List[Dict[str, Any]], prefix: str = "") -> Dict[str, float]:
    """
    Compute SwanLab metrics from tool_stats list.

    Args:
        tool_stats_list: List of tool_stats dictionaries
        prefix: Metric name prefix (e.g., "val/" for validation phase)

    Returns:
        Formatted metrics dictionary ready for SwanLab reporting
    """
    if not tool_stats_list:
        return {}

    metrics = {}

    # ========== 1. Overall Statistics ==========
    total_calls_list = [stats.get('total_calls', 0) for stats in tool_stats_list]
    success_calls_list = [stats.get('success_calls', 0) for stats in tool_stats_list]
    error_calls_list = [stats.get('total_errors', 0) for stats in tool_stats_list]
    cache_hits_list = [stats.get('cache_hits', 0) for stats in tool_stats_list]
    cache_misses_list = [stats.get('cache_misses', 0) for stats in tool_stats_list]

    # Calculate overall success rate
    total_calls_sum = sum(total_calls_list)
    success_calls_sum = sum(success_calls_list)
    tool_success_rate = (success_calls_sum / total_calls_sum * 100) if total_calls_sum > 0 else 0.0

    # Calculate overall cache hit rate
    cache_total = sum(cache_hits_list) + sum(cache_misses_list)
    cache_hit_rate = (sum(cache_hits_list) / cache_total * 100) if cache_total > 0 else 0.0

    metrics.update({
        f"{prefix}tool_stats/tool_success_rate": tool_success_rate,
        f"{prefix}tool_stats/tool_total_calls": float(np.mean(total_calls_list)),
        f"{prefix}tool_stats/tool_success_calls": float(np.mean(success_calls_list)),
        f"{prefix}tool_stats/tool_error_calls": float(np.mean(error_calls_list)),
        f"{prefix}tool_stats/tool_cache_hit_rate": cache_hit_rate,
        f"{prefix}tool_stats/tool_cache_hits": float(np.mean(cache_hits_list)),
        f"{prefix}tool_stats/tool_cache_misses": float(np.mean(cache_misses_list)),
    })

    # ========== 2. Time Consumption Statistics by Tool ==========
    tool_time_by_name = {}
    for stats in tool_stats_list:
        tool_time_dict = stats.get('tool_time', {})
        for tool_name, time_list in tool_time_dict.items():
            if tool_name not in tool_time_by_name:
                tool_time_by_name[tool_name] = []
            if isinstance(time_list, list):
                tool_time_by_name[tool_name].extend(time_list)

    for tool_name, time_list in tool_time_by_name.items():
        if time_list:
            metrics[f"{prefix}tool_time/{tool_name}/mean"] = float(np.mean(time_list))
            metrics[f"{prefix}tool_time/{tool_name}/max"] = float(np.max(time_list))
            metrics[f"{prefix}tool_time/{tool_name}/count"] = len(time_list)

    # ========== 3. Cache Hit Rate by Tool ==========
    tool_cache_by_name = {}
    for stats in tool_stats_list:
        tool_cache_stats = stats.get('tool_cache_stats', {})
        for tool_name, cache_info in tool_cache_stats.items():
            if tool_name not in tool_cache_by_name:
                tool_cache_by_name[tool_name] = {'hits': 0, 'misses': 0}
            tool_cache_by_name[tool_name]['hits'] += cache_info.get('hits', 0)
            tool_cache_by_name[tool_name]['misses'] += cache_info.get('misses', 0)

    for tool_name, cache_info in tool_cache_by_name.items():
        hits = cache_info['hits']
        misses = cache_info['misses']
        total = hits + misses
        if total > 0:
            hit_rate = hits / total * 100
            metrics[f"{prefix}tool_cache/{tool_name}/hit_rate"] = round(hit_rate, 2)
            metrics[f"{prefix}tool_cache/{tool_name}/hits"] = hits
            metrics[f"{prefix}tool_cache/{tool_name}/misses"] = misses

    # ========== 4. Error Rate by Tool ==========
    tool_error_by_name = {}
    for stats in tool_stats_list:
        tool_error_stats = stats.get('tool_error_stats', {})
        for tool_name, error_info in tool_error_stats.items():
            if tool_name not in tool_error_by_name:
                tool_error_by_name[tool_name] = {'calls': 0, 'errors': 0}
            tool_error_by_name[tool_name]['calls'] += error_info.get('calls', 0)
            tool_error_by_name[tool_name]['errors'] += error_info.get('errors', 0)

    for tool_name, error_info in tool_error_by_name.items():
        calls = error_info['calls']
        errors = error_info['errors']
        if calls > 0:
            error_rate = errors / calls * 100
            metrics[f"{prefix}tool_error/{tool_name}/error_rate"] = round(error_rate, 2)
            metrics[f"{prefix}tool_error/{tool_name}/calls"] = calls
            metrics[f"{prefix}tool_error/{tool_name}/errors"] = errors

    return metrics


def compute_tool_metrics_from_trajectories(trajectories: List[Any], prefix: str = "") -> Dict[str, float]:
    """
    Training phase: Extract tool_stats from trajectories and compute metrics.
    """
    tool_stats_list = extract_tool_stats_from_trajectories(trajectories)
    return compute_tool_metrics(tool_stats_list, prefix=prefix)


