# example_cocktail_rl_v2

Cocktail RL on AppWorld + AIME with configurable per-client batch ratios and an optional dynamic schedule.

```bash
source .venv/bin/activate && ajet-swarm start

# Export the SAME COCKTAIL_RATIO_SCHEDULE in both shells (linear | cos | constant).
export COCKTAIL_RATIO_SCHEDULE=constant
source .venv/bin/activate
python -m tutorial.example_cocktail_rl_v2.train_appworld_as_swarm_client_0   # driver
python -m tutorial.example_cocktail_rl_v2.train_aime_as_swarm_client_1       # follower
```

Edit `CocktailV2Config` defaults (cocktail_v2_runner.py) for `total_batch_size`, `schedule_start`/`schedule_end`/`schedule_end_step`. Engine knobs live in `build_cocktail_ajet_job()` (train_appworld_as_swarm_client_0.py). Both clients must agree on these.
