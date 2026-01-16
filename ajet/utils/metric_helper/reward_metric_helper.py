"""
FinWorld Reward Metrics Helper

Provides standalone utility functions for reward_stats extraction and SwanLab metrics formatting.
Decouples finworld-specific logic from core code, reducing intrusion into native_compat_trainer.

SwanLab metrics directory structure:
- rewards/                    Top-level aggregated scores
- rewards/dimensions/         Raw scores (unweighted)
- rewards/contribution/       Weighted contributions
- judge_time/                 Judge time consumption statistics
"""

from typing import List, Dict, Any
import numpy as np


def extract_reward_stats_from_trajectories(trajectories: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract reward_stats from trajectories list.

    Args:
        trajectories: List of trajectory objects containing workflow_metadata

    Returns:
        List of reward_stats dictionaries
    """
    reward_stats_list = []
    for traj in trajectories:
        if hasattr(traj, 'workflow_metadata') and traj.workflow_metadata:
            if 'reward_stats' in traj.workflow_metadata:
                reward_stats_list.append(traj.workflow_metadata['reward_stats'])
    return reward_stats_list


def extract_reward_stats_from_cmts(cmts: List[Any]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract reward_stats from cmts list and return debug statistics.

    Args:
        cmts: List of cmt objects containing workflow_metadata

    Returns:
        Tuple of (reward_stats_list, debug_stats)
    """
    reward_stats_list = []
    debug_stats = {
        'total_cmts': len(cmts),
        'has_workflow_metadata': 0,
        'has_reward_stats': 0,
    }

    for _cmt in cmts:
        if hasattr(_cmt, 'workflow_metadata') and _cmt.workflow_metadata:
            debug_stats['has_workflow_metadata'] += 1
            if 'reward_stats' in _cmt.workflow_metadata:
                debug_stats['has_reward_stats'] += 1
                reward_stats_list.append(_cmt.workflow_metadata['reward_stats'])

    return reward_stats_list, debug_stats


def compute_reward_metrics(reward_stats_list: List[Dict[str, Any]], prefix: str = "") -> Dict[str, float]:
    """
    Compute SwanLab metrics from reward_stats list.

    Supports two data sources:
    1. RM Gallery RewardStats fields (rm_raw, etc.)
    2. OpenJudge fields (openjudge_xxx_raw, openjudge_xxx_contribution, etc.)

    Args:
        reward_stats_list: List of reward_stats dictionaries
        prefix: Metric name prefix (e.g., "val/" for validation phase)

    Returns:
        Formatted metrics dictionary ready for SwanLab reporting
    """
    if not reward_stats_list:
        return {}

    n = len(reward_stats_list)
    metrics = {}

    # ========== Top-level Scores (General) ==========
    final_reward_list = [rs.get('final_reward', 0.0) for rs in reward_stats_list]
    fused_reward_list = [rs.get('fused_reward', 0.0) for rs in reward_stats_list]
    penalty_list = [rs.get('penalty', 0.0) for rs in reward_stats_list]
    step_reward_list = [rs.get('step_reward', 0.0) for rs in reward_stats_list]

    # Penalty statistics
    non_zero_penalties = [p for p in penalty_list if p != 0.0]

    # Top-level metrics
    metrics[f"{prefix}rewards/final_reward_mean"] = float(np.mean(final_reward_list))
    metrics[f"{prefix}rewards/fused_reward_mean"] = float(np.mean(fused_reward_list))
    metrics[f"{prefix}rewards/penalty_mean"] = float(np.mean(penalty_list))
    metrics[f"{prefix}rewards/step_reward_mean"] = float(np.mean(step_reward_list))
    metrics[f"{prefix}rewards/penalty_count"] = len(non_zero_penalties)
    metrics[f"{prefix}rewards/penalty_rate"] = len(non_zero_penalties) / n * 100 if n > 0 else 0.0

    # ========== Detect OpenJudge Usage ==========
    openjudge_enabled_count = sum(1 for rs in reward_stats_list if rs.get('openjudge_enabled', False))

    if openjudge_enabled_count > 0:
        # ========== OpenJudge Metrics ==========
        metrics[f"{prefix}rewards/openjudge_enabled_rate"] = openjudge_enabled_count / n * 100

        # Dynamically extract OpenJudge grader fields
        # Currently supported graders: report_resolution, trajectory_faithfulness,
        # rubrics_performance, trajectory_comprehensive, information_gain, action_loop
        openjudge_graders = [
            "report_resolution",
            "trajectory_faithfulness",
            "rubrics_performance",
            "trajectory_comprehensive",
            "information_gain",
            "action_loop",
        ]

        for grader_name in openjudge_graders:
            raw_key = f"openjudge_{grader_name}_raw"
            contrib_key = f"openjudge_{grader_name}_contribution"

            raw_list = [rs.get(raw_key, 0.0) for rs in reward_stats_list]
            contrib_list = [rs.get(contrib_key, 0.0) for rs in reward_stats_list]

            # Only report when non-zero values exist
            if any(v != 0.0 for v in raw_list):
                metrics[f"{prefix}rewards/openjudge/{grader_name}_raw_mean"] = float(np.mean(raw_list))
            if any(v != 0.0 for v in contrib_list):
                metrics[f"{prefix}rewards/openjudge/{grader_name}_contribution_mean"] = float(np.mean(contrib_list))

        # OpenJudge time consumption statistics
        grading_time_list = [rs.get('grading_time', 0.0) for rs in reward_stats_list]
        if any(v != 0.0 for v in grading_time_list):
            metrics[f"{prefix}judge_time/openjudge_grading_time_mean"] = float(np.mean(grading_time_list))
            metrics[f"{prefix}judge_time/openjudge_grading_time_max"] = float(np.max(grading_time_list))

    # ========== RM Gallery Metrics ==========

    # RM Gallery
    rm_raw_list = [rs.get('rm_raw', 0.0) for rs in reward_stats_list]
    rm_contribution_list = [rs.get('rm_contribution', 0.0) for rs in reward_stats_list]

    # RefJudge
    ref_final_raw_list = [rs.get('ref_final_raw', 0.0) for rs in reward_stats_list]
    ref_citation_raw_list = [rs.get('ref_citation_raw', 0.0) for rs in reward_stats_list]
    ref_grounding_raw_list = [rs.get('ref_grounding_raw', 0.0) for rs in reward_stats_list]
    ref_contribution_list = [rs.get('ref_contribution', 0.0) for rs in reward_stats_list]

    # StructureJudge
    structure_raw_list = [rs.get('structure_raw', 0.0) for rs in reward_stats_list]
    structure_contribution_list = [rs.get('structure_contribution', 0.0) for rs in reward_stats_list]

    # dimensions/ raw scores
    metrics[f"{prefix}rewards/dimensions/rm_raw_mean"] = float(np.mean(rm_raw_list))
    metrics[f"{prefix}rewards/dimensions/ref_final_raw_mean"] = float(np.mean(ref_final_raw_list))
    metrics[f"{prefix}rewards/dimensions/ref_citation_raw_mean"] = float(np.mean(ref_citation_raw_list))
    metrics[f"{prefix}rewards/dimensions/ref_grounding_raw_mean"] = float(np.mean(ref_grounding_raw_list))
    metrics[f"{prefix}rewards/dimensions/structure_raw_mean"] = float(np.mean(structure_raw_list))

    # contribution/ weighted contributions
    metrics[f"{prefix}rewards/contribution/rm_contribution_mean"] = float(np.mean(rm_contribution_list))
    metrics[f"{prefix}rewards/contribution/ref_contribution_mean"] = float(np.mean(ref_contribution_list))
    metrics[f"{prefix}rewards/contribution/structure_contribution_mean"] = float(np.mean(structure_contribution_list))

    # Enabled state statistics
    ref_judge_enabled_count = sum(1 for rs in reward_stats_list if rs.get('ref_judge_enabled', False))
    if ref_judge_enabled_count > 0:
        metrics[f"{prefix}rewards/ref_judge_enabled_rate"] = ref_judge_enabled_count / n * 100

    structure_judge_enabled_count = sum(1 for rs in reward_stats_list if rs.get('structure_judge_enabled', False))
    if structure_judge_enabled_count > 0:
        metrics[f"{prefix}rewards/structure_judge_enabled_rate"] = structure_judge_enabled_count / n * 100

    # Time consumption statistics
    rm_time_list = [rs.get('rm_time', 0.0) for rs in reward_stats_list]
    refstruc_time_list = [rs.get('refstruc_time', 0.0) for rs in reward_stats_list]

    metrics[f"{prefix}judge_time/rm_time_mean"] = float(np.mean(rm_time_list))
    metrics[f"{prefix}judge_time/refstruc_time_mean"] = float(np.mean(refstruc_time_list))

    if rm_time_list:
        metrics[f"{prefix}judge_time/rm_time_max"] = float(np.max(rm_time_list))
    if refstruc_time_list:
        metrics[f"{prefix}judge_time/refstruc_time_max"] = float(np.max(refstruc_time_list))

    # ========== General Time Consumption Statistics ==========
    judge_total_time_list = [rs.get('judge_total_time', 0.0) for rs in reward_stats_list]
    if any(v != 0.0 for v in judge_total_time_list):
        metrics[f"{prefix}judge_time/judge_total_time_mean"] = float(np.mean(judge_total_time_list))
        metrics[f"{prefix}judge_time/judge_total_time_max"] = float(np.max(judge_total_time_list))

    return metrics


def compute_reward_metrics_from_trajectories(trajectories: List[Any]) -> Dict[str, float]:
    """
    Training phase: Extract reward_stats from trajectories and compute metrics.

    Args:
        trajectories: List of trajectory objects

    Returns:
        Formatted metrics dictionary
    """
    reward_stats_list = extract_reward_stats_from_trajectories(trajectories)
    return compute_reward_metrics(reward_stats_list, prefix="train_")


def compute_reward_metrics_from_cmts(cmts: List[Any], print_debug: bool = True) -> Dict[str, float]:
    """
    Validation phase: Extract reward_stats from cmts and compute metrics.

    Args:
        cmts: List of cmt objects
        print_debug: Whether to print debug information

    Returns:
        Formatted metrics dictionary (with "val_reward/" prefix)
    """
    reward_stats_list, debug_stats = extract_reward_stats_from_cmts(cmts)

    if print_debug:
        print(f"\n[DEBUG eval_dataset()] reward_stats statistics:")
        print(f"  - Total cmts count: {debug_stats['total_cmts']}")
        print(f"  - Has workflow_metadata: {debug_stats['has_workflow_metadata']}")
        print(f"  - Has reward_stats: {debug_stats['has_reward_stats']}")
        print(f"  - Extracted samples count: {len(reward_stats_list)}")

    return compute_reward_metrics(reward_stats_list, prefix="val_")
