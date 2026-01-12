import importlib
import importlib.util
import os
import sys
import threading
from typing import Any, Callable, Union


def cls_to_path(obj_or_path: Union[str, Callable[..., Any]]) -> str:
    """Convert a callable to the ``module->name`` string expected by dynamic_import."""

    if isinstance(obj_or_path, str):
        return obj_or_path
    module = getattr(obj_or_path, "__module__", None)
    name = getattr(obj_or_path, "__name__", None)
    if module and name:
        return f"{module}->{name}"
    raise ValueError("Object must be a dotted string or a callable with __module__ and __name__.")


def _dynamic_import(module_class_str: str):
    """
    Dynamic import of class from module
    Supports two formats:
    1. module.path->ClassName (dot-separated module path)
    2. path/to/module.py->ClassName (file path format, can be absolute or relative)
    """
    module_str, class_name = module_class_str.split("->")

    # Use .py-> as identifier for file path format
    if ".py->" in module_class_str:
        # Handle file path format
        file_path = module_str

        # Convert to absolute path
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise ImportError(f"Module file not found: {file_path}")

        # Split module name
        module_name = os.path.splitext(os.path.basename(file_path))[0]

        # Load module from file using importlib.util
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for: {file_path}")

        module = importlib.util.module_from_spec(spec)

        # 检查是否已经加载过这个模块
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            # Load module from file using importlib.util
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot create module spec for: {file_path}")

            module = importlib.util.module_from_spec(spec)

            # Add module to sys.modules BEFORE execution to avoid duplicate loading
            sys.modules[module_name] = module

            # Execute module
            spec.loader.exec_module(module)

    else:
        # Standard module path format
        module = importlib.import_module(module_str)

    # Get class
    try:
        protocol_cls = getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Cannot import class {class_name} from module {module_str}: {e}") from e
    return protocol_cls


_import_lock = threading.RLock()


def dynamic_import(module_class_str: str):
    """
    Thread-safe dynamic import of a class from a module string.
    Args:
        module_class_str (str): Module and class string, e.g., 'module.path->ClassName' or 'path/to/module.py->ClassName'.
    Returns:
        type: The imported class type.
    """
    with _import_lock:
        return _dynamic_import(module_class_str)
