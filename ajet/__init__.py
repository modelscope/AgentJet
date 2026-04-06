__version__ = "0.1.0"

__all__ = [
    "Workflow",
    "WorkflowTask",
    "WorkflowOutput",
    "AjetTuner",
    "AgentJetJob",
    "bp"
]

_LAZY_IMPORTS = {
    "AjetTuner": "ajet.tuner",
    "AgentJetJob": "ajet.copilot.job",
    "WorkflowOutput": "ajet.schema.task",
    "WorkflowTask": "ajet.schema.task",
    "Workflow": "ajet.workflow",
    "bp": "ajet.utils.vsdb",
}

_ATTR_MAPPING = {
    "bp": "vscode_conditional_breakpoint"
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)

        attr_name = _ATTR_MAPPING.get(name, name)
        value = getattr(module, attr_name)  # type: ignore

        globals()[name] = value
        return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
