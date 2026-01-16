# from typing import Any, Dict, List
# from loguru import logger
# from omegaconf import DictConfig
# from openai.types.chat.chat_completion import ChatCompletion
# from verl import DataProto


from pydantic import BaseModel


class TokenAndProb(BaseModel):
    token_id: int
    logprob: float
    decoded_string: str
