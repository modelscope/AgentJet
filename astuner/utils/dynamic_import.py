import importlib
import importlib.util
import os
import sys
import threading


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
    protocol_cls = getattr(module, class_name)
    return protocol_cls


_import_lock = threading.RLock()


def dynamic_import(module_class_str: str):
    with _import_lock:
        return _dynamic_import(module_class_str)
