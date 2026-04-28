# -*- coding: utf-8 -*-
"""
AIME Math Swarm Training Module

This module provides a complete training pipeline for math problem-solving
using the DAPO-Math-17k dataset and AgentJet Swarm.

Usage:
    # Download datasets
    proxychains python -m tutorial.opencode_build_aime.download_data

    # Start training
    python -m tutorial.opencode_build_aime.agent_roll
"""

from tutorial.opencode_build_aime.agent_run_v3 import (
    execute_agent,
    run_agent_and_compute_reward,
    compute_reward,
)

__all__ = [
    "execute_agent",
    "run_agent_and_compute_reward",
    "compute_reward",
]
