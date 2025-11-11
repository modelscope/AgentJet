from astune.context_manager.cmt_linear import CMTLinear, ExtendedMessage
from astune.context_manager.cmt_linear_think import LinearThinkCMT
from typing import Any, Dict, List, Union, Callable

class BaseAgentFlow(object):

    def __init__(self,
                 llm_chat_fn: Callable,
                 tokenizer: Any,
                 config,
                 **kwargs):
        self.tokenizer = tokenizer
        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.cmt: Union[CMTLinear, LinearThinkCMT, Any, None] = None
        self.alien_llm_chat_fn: Union[Callable, None] = None
        self.llm_chat_fn: Callable = llm_chat_fn
        self.config = config
        # self.console_debug_mode: bool = False
        self.max_steps: int = self.config.astune.rollout.multi_turn.max_steps
        self.max_model_len: int = self.config.astune.rollout.max_model_len
        self.max_env_len: int = self.config.astune.rollout.max_env_len