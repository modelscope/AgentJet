
import importlib

def dynamic_import(module_class_str: str):
    module_, class_ = module_class_str.split('->')
    protocol_cls = getattr(importlib.import_module(module_), class_)
    return protocol_cls
