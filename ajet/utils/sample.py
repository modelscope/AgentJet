def get_sample_params(mode, config):
    """
    Generate sampling parameters for text generation based on mode and config.
    Args:
        mode (str): The mode of operation, e.g., 'validate'.
        config: Configuration object containing rollout parameters.
    Returns:
        dict: Sampling parameters for the model.
    """
    response_length_eps = (
        16  # Reserve a few tokens for later handling of special tokens like lm_start.
    )
    if config.ajet.rollout.name == "vllm":
        # VLLM uses max_tokens instead of max_new_tokens
        sampling_params = dict(
            n=1,
            max_tokens=config.ajet.rollout.max_response_length_in_one_turn - response_length_eps,
            min_tokens=1,  # Must output at least 1 token.
            temperature=config.ajet.rollout.temperature,
            top_p=config.ajet.rollout.top_p,
            logprobs=1,
        )
    else:
        sampling_params = dict(
            n=1,
            max_new_tokens=config.ajet.rollout.max_response_length_in_one_turn,
            temperature=config.ajet.rollout.temperature,
            top_p=config.ajet.rollout.top_p,
        )

    if mode == "validate":
        sampling_params["temperature"] = config.ajet.rollout.val_kwargs.temperature
        sampling_params["top_k"] = config.ajet.rollout.val_kwargs.top_k
        sampling_params["top_p"] = config.ajet.rollout.val_kwargs.top_p
    return sampling_params
