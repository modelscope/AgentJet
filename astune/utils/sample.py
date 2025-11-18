


def get_sample_params(mode, config):
    response_length_eps = 16  # Reserve a few tokens for later handling of special tokens like lm_start.
    if config.astune.rollout.name == 'vllm':
        sampling_params = dict(
            n=1,
            max_tokens=config.astune.rollout.max_response_length_in_one_turn - response_length_eps,
            min_tokens=1,   # Must output at least 1 token.
            temperature=config.astune.rollout.temperature,
            top_p=config.astune.rollout.top_p
        )
    else:
        sampling_params = dict(
            n=1,
            max_new_tokens=config.astune.rollout.max_response_length_in_one_turn,
            temperature=config.astune.rollout.temperature,
            top_p=config.astune.rollout.top_p
        )

    if mode == "validate":
        sampling_params["temperature"] = config.astune.rollout.val_kwargs.temperature
        sampling_params["top_k"] = config.astune.rollout.val_kwargs.top_k
        sampling_params["top_p"] = config.astune.rollout.val_kwargs.top_p
    return sampling_params
