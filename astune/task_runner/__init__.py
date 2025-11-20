from astune.context_tracker.basic_tracker import BasicContextTracker
from typing import Any, Dict, List, Union, Callable

class BaseAgentRunner(object):

    def __init__(self,
                 llm_chat_fn: Callable,
                 tokenizer: Any,
                 config,
                 **kwargs):
        self.tokenizer = tokenizer
        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.cmt: Union[BasicContextTracker, Any, None] = None
        self.alien_llm_chat_fn: Union[Callable, None] = None
        self.llm_chat_fn: Callable = llm_chat_fn
        self.config = config
        self.max_steps: int = self.config.astune.rollout.multi_turn.max_steps
        self.max_model_len: int = self.config.astune.rollout.max_model_len
        self.max_env_len: int = self.config.astune.rollout.max_env_len
