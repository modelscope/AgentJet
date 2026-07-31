"""
Compute the blackout token combo for loss masking.

The blackout combo is the leading token sequence of the generation-prompt
header that must NOT be trained on. It is selected by config so different chat
templates can black out different headers.

Config key: ``ajet.rollout.chat_template_generate_prompt_type``
  - "default" : ``<|im_start|>assistant\\n``          (standard Qwen header)
  - "qwen3.6" : ``<|im_start|>assistant\\n<think>``    (also blacks out the
                 reasoning-scaffold <think> opener the Qwen3.6 template
                 injects into the generation prompt)

Note the "qwen3.6" header deliberately stops at ``<think>`` and does NOT include
the trailing newline. That newline (token 198) merges with following content
into a single ``\\n\\n`` token (271) when the message has no leading-newline
content, which would break exact sublist matching. Stopping at ``<think>``
keeps the header merge-stable.
"""

from transformers.tokenization_utils import PreTrainedTokenizer


# Header strings keyed by chat_template_generate_prompt_type.
_HEADER_BY_PROMPT_TYPE = {
    "default": "<|im_start|>assistant\n",
    "qwen3.6": "<|im_start|>assistant\n<think>",
}


def get_blackout_token_combo(tokenizer: PreTrainedTokenizer, config=None) -> list:
    """Return the token ids of the generation-prompt header to black out.

    Args:
        tokenizer: The model tokenizer.
        config: The ajet config object (OmegaConf/dataclass). When provided,
            ``config.ajet.rollout.chat_template_generate_prompt_type`` selects
            the header. Missing/None config falls back to the default header.

    Returns:
        List of token ids for the header string.
    """
    prompt_type = "default"
    if config is not None:
        try:
            prompt_type = config.ajet.rollout.chat_template_generate_prompt_type or "default"
        except (AttributeError, KeyError):
            prompt_type = "default"

    header = _HEADER_BY_PROMPT_TYPE.get(prompt_type)
    if header is None:
        raise ValueError(
            "Unknown ajet.rollout.chat_template_generate_prompt_type="
            f"{prompt_type!r}. Supported: {sorted(_HEADER_BY_PROMPT_TYPE)}"
        )

    return tokenizer.encode(header)
