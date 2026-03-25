#!/usr/bin/env python3
"""
Script to remove _target_ fields from YAML configuration files.
Before removing, validates that all parameters in the target class exist in the YAML config.
Also adds comments to entries that appear in the auto-conversion config.
"""

import yaml
import importlib
import inspect
import json
import re
from loguru import logger
from typing import Any, Dict, Set, List
from pathlib import Path
import sys


class TargetRemovalError(Exception):
    """Custom exception for target removal validation errors."""
    pass


def get_class_from_target(target_path: str):
    """
    Import and return a class from a dotted path like 'verl.workers.config.FSDPOptimizerConfig'.

    Args:
        target_path: Dotted path to the class

    Returns:
        The imported class object
    """
    module_path, class_name = target_path.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        raise TargetRemovalError(f"Failed to import {target_path}: {e}")


def get_class_parameters(cls) -> Set[str]:
    """
    Get all parameter names from a class's __init__ method.

    Args:
        cls: The class to inspect

    Returns:
        Set of parameter names (excluding 'self')
    """
    try:
        sig = inspect.signature(cls.__init__)
        params = set(sig.parameters.keys())
        params.discard('self')
        # Also check for dataclass fields
        if hasattr(cls, '__dataclass_fields__'):
            params.update(cls.__dataclass_fields__.keys())
        return params
    except Exception as e:
        raise TargetRemovalError(f"Failed to inspect class {cls.__name__}: {e}")


def get_config_keys(config: Any) -> Set[str]:
    """
    Get all keys from a configuration dict, excluding special keys like _target_.

    Args:
        config: Configuration dict or value

    Returns:
        Set of configuration keys
    """
    if not isinstance(config, dict):
        return set()

    keys = set(config.keys())
    keys.discard('_target_')
    return keys


def parse_jsonc(file_path: str) -> Dict[str, Any]:
    """Parse a JSONC file (JSON with comments)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove comments (both // and /* */ style)
    content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    return json.loads(content)


def get_all_yaml_paths(data: Any, prefix: str = "") -> Set[str]:
    """
    Recursively extract all key paths from a YAML structure.
    Returns paths like 'actor_rollout_ref.actor.optim.lr'
    """
    paths = set()

    if isinstance(data, dict):
        for key, value in data.items():
            if key == '_target_':  # Skip _target_ keys
                continue

            current_path = f"{prefix}.{key}" if prefix else key
            paths.add(current_path)

            # Recursively process nested structures
            if isinstance(value, (dict, list)):
                paths.update(get_all_yaml_paths(value, current_path))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                paths.update(get_all_yaml_paths(item, prefix))

    return paths


def get_conversion_targets(conversion_config: Dict[str, Any]) -> Set[str]:
    """
    Extract all target paths from the conversion config.
    Returns a set of paths like 'actor_rollout_ref.actor.optim'
    """
    targets = set()

    for key, value in conversion_config.items():
        if isinstance(value, str):
            targets.add(value)
        elif isinstance(value, list):
            targets.update(value)

    return targets


def add_comments_to_yaml_lines(yaml_lines: List[str], yaml_data: Dict, conversion_targets: Set[str]) -> List[str]:
    """
    Add comments to YAML lines that match conversion targets.
    """
    result_lines = []
    path_stack = []
    indent_stack = []

    for line in yaml_lines:
        # Calculate indentation
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            result_lines.append(line)
            continue

        indent = len(line) - len(stripped)

        # Update path stack based on indentation
        while indent_stack and indent <= indent_stack[-1]:
            indent_stack.pop()
            if path_stack:
                path_stack.pop()

        # Extract key from line
        if ':' in stripped:
            key = stripped.split(':')[0].strip()

            # Skip special keys
            if key in ['_target_', '-']:
                result_lines.append(line)
                continue

            # Build current path
            path_stack.append(key)
            indent_stack.append(indent)

            current_path = '.'.join(path_stack)

            # Check if this path is in conversion targets
            if current_path in conversion_targets:
                # Add comment if not already present
                if '# [auto-convert]' not in line:
                    line = line.rstrip() + '  # [auto-convert]\n'

            result_lines.append(line)
        else:
            result_lines.append(line)

    return result_lines

def validate_and_remove_targets(data: Any, path: str = "root") -> Any:

    if isinstance(data, dict):
        data = {key: validate_and_remove_targets(value, f"{path}.{key}") for key, value in data.items()}

        if '_target_' not in data:
            return data
        target_path = data['_target_']

        # Import the target class
        target_class = get_class_from_target(target_path)

        # Get parameters from the class
        class_params = get_class_parameters(target_class)

        # Get keys from current config (excluding _target_)
        config_keys = get_config_keys(data)

        # Check if there are any class parameters missing in config
        # Note: It's OK for config to have extra keys (like nested configs)
        # We're checking if class has required params that aren't in config
        missing_in_config = class_params - config_keys - {"_target_"}
        extra_in_config = config_keys - class_params - {"_target_"}

        if extra_in_config.__len__ != 0:
            for k in extra_in_config:
                logger.error(f"Error: discovered unidentified config: {path}.{k}")

        sig = inspect.signature(target_class.__init__)
        params_with_defaults = {
            name: param for name, param in sig.parameters.items()
            if param.default != inspect.Parameter.empty
        }
        for key in missing_in_config:
            if str(params_with_defaults[key].default) != '<factory>':
                # add to data
                print(f"[{path}] add {key} with default value {params_with_defaults[key]} from class {target_class}")
                data[key] = params_with_defaults[key].default
            else:
                # str(params_with_defaults['engine']._annotation)
                # target_instance = target_class(**data)
                # if isinstance(getattr(target_instance, key), dict):
                #     data[key] = getattr(target_instance, key)
                # else:
                #     data[key] = {
                #         '_target_': str(getattr(target_instance, key).__class__).split("'")[1]
                #     }
                if "<class" not in str(params_with_defaults[key]._annotation):
                    logger.warning(f"warning! cannot process {path}.{key}!!")
                else:
                    class_str = str(params_with_defaults[key]._annotation).split("'")[1]
                    if class_str == 'dict':
                        data[key] = {}
                    else:
                        data[key] = {
                            '_target_': class_str
                        }
                        data[key] = validate_and_remove_targets(data[key], f"{path}.{key}")

        return data

    elif isinstance(data, list):
        data = [validate_and_remove_targets(item, f"{path}[{i}]") for i, item in enumerate(data)]
        return data
    else:
        return data


def remove_targets_from_yaml(input_file: Path, output_file: Path | None = None, validate: bool = True, add_conversion_comments: bool = True):
    """
    Remove _target_ fields from a YAML configuration file.
    Optionally add comments to entries that appear in the auto-conversion config.

    Args:
        input_file: Path to input YAML file
        output_file: Path to output YAML file (defaults to overwriting input)
        validate: Whether to validate target classes before removal
        add_conversion_comments: Whether to add comments for auto-converted fields

    Raises:
        TargetRemovalError: If validation fails
    """
    # Read the YAML file
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)

    if not validate:
        print("⚠ Warning: Validation is disabled. Removing all _target_ fields without checks.")

    # Process the data
    processed_data = validate_and_remove_targets(data)

    # Write back to file
    output_path = output_file or input_file

    if add_conversion_comments:
        # Try to load conversion config
        conversion_config_path = Path("ajet/default_config/verl/config_auto_convertion_verl.jsonc")
        if conversion_config_path.exists():
            print(f"\nLoading conversion config from {conversion_config_path}")
            try:
                conversion_config = parse_jsonc(str(conversion_config_path))
                conversion_targets = get_conversion_targets(conversion_config)

                # First write YAML without comments
                with open(output_path, 'w') as f:
                    yaml.dump(processed_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

                # Then add comments
                with open(output_path, 'r', encoding='utf-8') as f:
                    yaml_lines = f.readlines()

                # Use unsafe loader to handle Python tuples
                with open(output_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.unsafe_load(f)

                # Get all YAML paths
                yaml_paths = get_all_yaml_paths(yaml_data)

                # Find matches
                matches = yaml_paths & conversion_targets
                print(f"Found {len(matches)} entries matching conversion config")

                # Add comments to matching lines
                result_lines = add_comments_to_yaml_lines(yaml_lines, yaml_data, conversion_targets)

                # Write final output with comments
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.writelines(result_lines)

                print(f"Added # [auto-convert] comments to {len(matches)} entries")
            except Exception as e:
                print(f"⚠ Warning: Failed to add conversion comments: {e}")
                # Write without comments
                with open(output_path, 'w') as f:
                    yaml.dump(processed_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        else:
            print(f"⚠ Warning: Conversion config not found at {conversion_config_path}")
            # Write without comments
            with open(output_path, 'w') as f:
                yaml.dump(processed_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    else:
        # Write without comments
        with open(output_path, 'w') as f:
            yaml.dump(processed_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\n✓ Successfully processed {input_file}")
    if output_file:
        print(f"  Output written to: {output_file}")
    else:
        print(f"  File updated in place")


def main():
    import os

    # Parse command-line arguments
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
Usage: python remove_targets.py [OPTIONS]

Remove _target_ fields from YAML configuration files and optionally add
comments for auto-converted fields.

Options:
  --no-validate      Skip validation of target classes
  --no-comments      Skip adding auto-conversion comments
  -h, --help         Show this help message

Input:  ajet/default_config/verl/verl_default.yaml
Output: ajet/default_config/verl/verl_default_no_targets.yaml

Features:
  - Validates and removes _target_ fields from YAML
  - Adds missing default parameters from target classes
  - Marks auto-converted fields with # [auto-convert] comments
""")
        sys.exit(0)

    input_file = Path("ajet/default_config/verl/verl_default.yaml")
    output_file = Path("ajet/default_config/verl/verl_default_expand.yaml")
    validate = '--no-validate' not in sys.argv
    add_comments = '--no-comments' not in sys.argv

    assert os.path.exists(input_file), f"Input file not found: {input_file}"

    remove_targets_from_yaml(input_file, output_file, validate, add_comments)


if __name__ == '__main__':
    main()
