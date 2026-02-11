__all__ = [
    "AgentScopeModelTuner",
    "OpenaiClientModelTuner",
]

_LAZY_IMPORTS = {
    "AgentScopeModelTuner": "ajet.tuner_lib.as_agentscope_model",
    "OpenaiClientModelTuner": "ajet.tuner_lib.as_oai_sdk_model",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)

        value = getattr(module, name)

        globals()[name] = value
        return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
