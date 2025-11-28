def get_sample_params(mode, config):
    response_length_eps = (
        16  # Reserve a few tokens for later handling of special tokens like lm_start.
    )
    if config.astuner.rollout.name == "vllm":
        sampling_params = dict(
            n=1,
            max_tokens=config.astuner.rollout.max_response_length_in_one_turn - response_length_eps,
            min_tokens=1,  # Must output at least 1 token.
            temperature=config.astuner.rollout.temperature,
            top_p=config.astuner.rollout.top_p,
        )
    else:
        sampling_params = dict(
            n=1,
            max_new_tokens=config.astuner.rollout.max_response_length_in_one_turn,
            temperature=config.astuner.rollout.temperature,
            top_p=config.astuner.rollout.top_p,
        )

    if mode == "validate":
        sampling_params["temperature"] = config.astuner.rollout.val_kwargs.temperature
        sampling_params["top_k"] = config.astuner.rollout.val_kwargs.top_k
        sampling_params["top_p"] = config.astuner.rollout.val_kwargs.top_p
    return sampling_params
