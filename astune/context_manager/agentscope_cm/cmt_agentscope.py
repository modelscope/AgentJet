import importlib
from loguru import logger

from agentscope._utils._common import _json_loads_with_repair, _create_tool_from_base_model
from astune.context_manager.agentscope_cm.cmt_request_proxy import ASTuneLmProxy

from typing import Any, List, Dict


class ASTuneProxy(ASTuneLmProxy):
    """
    A proxy class that bridge:
    - environment
    - reward
    - policy llm model
    """


    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        tool_choice = None,
        structured_model = None,
        **kwargs: Any,
    ):

        # For qvq and qwen-vl models, the content field cannot be `None` or
        # `[{"text": None}]`, so we need to convert it to an empty list.
        if self.model_name.startswith("qvq") or "-vl" in self.model_name:
            raise NotImplementedError("Not implemented for qvq and qwen-vl models yet.")

        kwargs = {
            "messages": messages,
            "model": self.model_name,
            "stream": self.stream,
            **self.dscm_ref.generate_kwargs,
            **kwargs,
            "result_format": "message",
            # In agentscope, the `incremental_output` must be `True` when
            # `self.stream` is True
            "incremental_output": self.stream,
        }

        if tools:
            kwargs["tools"] = self.dscm_ref._format_tools_json_schemas(tools)

        if tool_choice:
            self.dscm_ref._validate_tool_choice(tool_choice, tools)
            kwargs["tool_choice"] = self.dscm_ref._format_tool_choice(tool_choice)

        if (
            self.dscm_ref.enable_thinking is not None
            and "enable_thinking" not in kwargs
        ):
            kwargs["enable_thinking"] = self.dscm_ref.enable_thinking

        if structured_model:
            if tools or tool_choice:
                logger.warning(
                    "structured_model is provided. Both 'tools' and "
                    "'tool_choice' parameters will be overridden and "
                    "ignored. The model will only perform structured output "
                    "generation without calling any other tools.",
                )
            format_tool = _create_tool_from_base_model(structured_model)
            kwargs["tools"] = self.dscm_ref._format_tools_json_schemas(
                [format_tool],
            )
            kwargs["tool_choice"] = self.dscm_ref._format_tool_choice(
                format_tool["function"]["name"],
            )

        response = await self.execute_model_proxy(
            api_key=self.dscm_ref.api_key,
            structured_model=structured_model,
            **kwargs,
        )
        return response

    def update_agentscope_input_dictionary(self, **kwargs):
        self.input_kwargs.update(kwargs)

    def get_agentscope_input_dictionary(self):
        return self.input_kwargs

    def update_judge_input_dictionary(self, **kwargs):
        self.output_kwargs.update(kwargs)

    def get_judge_input_dictionary(self):
        return self.output_kwargs

    def get_judge(self):
        judge_protocol = self.config.astune.task_judge.judge_protocol
        module_, class_ = judge_protocol.split('->')
        protocol_cls = getattr(importlib.import_module(module_), class_)
        return protocol_cls(self.config)  # type: ignore
