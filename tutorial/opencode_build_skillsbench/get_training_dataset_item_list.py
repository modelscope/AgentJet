#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset collector for SkillsBench tasks.
Returns a list of training task identifiers.
"""

import os
from pathlib import Path
from typing import List, Dict


def get_training_dataset_item_list() -> List[Dict[str, str]]:
    """
    Get list of SkillsBench tasks for training.
    
    Returns:
        List of dicts, each containing task metadata:
        - task_id: unique identifier for the task
        - task_path: full path to the task directory
    """
    # Path to skillsbench repository
    skillsbench_root = Path("/root/AgentJet/tmp/skillsbench_swarm_test")
    tasks_dir = skillsbench_root / "tasks"
    
    if not tasks_dir.exists():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")
    
    task_list = []
    
    # Iterate through all task directories
    for task_path in sorted(tasks_dir.iterdir()):
        if not task_path.is_dir():
            continue
            
        task_id = task_path.name
        
        # Verify this is a valid task (has required files)
        instruction_file = task_path / "instruction.md"
        task_toml = task_path / "task.toml"
        tests_dir = task_path / "tests"
        
        if not (instruction_file.exists() and task_toml.exists() and tests_dir.exists()):
            print(f"Warning: Skipping invalid task: {task_id}")
            continue
        
        task_list.append({
            "task_id": task_id,
            "task_path": str(task_path),
        })
    
    print(f"Found {len(task_list)} valid tasks for training")
    return task_list


if __name__ == "__main__":
    # Test the function
    tasks = get_training_dataset_item_list()
    print(f"\nTotal tasks: {len(tasks)}")
    print("\nFirst 5 tasks:")
    for i, task in enumerate(tasks[:5]):
        print(f"{i+1}. {task['task_id']}")
