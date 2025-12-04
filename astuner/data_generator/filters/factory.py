from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from .base import Filter
from .llm_evaluate_filter import LlmEvaluateFilter
from .deduplication_filter import DeduplicationFilter

FILTER_REGISTRY: Dict[str, type[Filter]] = {
    "llm_evaluate": LlmEvaluateFilter,
    "deduplication": DeduplicationFilter,
}

def _build_single_filter(spec: Mapping[str, Any]) -> Filter:
    type_name = spec.get("type")
    if not isinstance(type_name, str):
        raise ValueError(f"Filter spec must contain string 'type', got: {type_name!r}")

    params = spec.get("params") or {}
    if not isinstance(params, MutableMapping):
        raise TypeError("Filter 'params' must be a mapping if present")

    cls = FILTER_REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown filter type: {type_name!r}")

    return cls(**params)  # type: ignore[arg-type]


def build_filters(specs: Sequence[Mapping[str, Any]] | None) -> List[Filter]:
    """Setup filter chain according to config.
    Refer to: astuner.task_reader.feedback_tracing.filters
    """
    if not specs:
        return []

    filters: List[Filter] = []
    for spec in specs:
        enabled = spec.get("enabled", True)
        if not enabled:
            continue
        filters.append(_build_single_filter(spec))
    return filters
