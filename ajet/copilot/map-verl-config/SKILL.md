---
name: map-verl-config
description: map verl config to agentjet config
license: Complete terms in LICENSE.txt
---


1. find user requested verl config in codebase/agentjet/ajet/default_config/verl/verl_default.yaml

2. check `codebase/agentjet/ajet/default_config/verl/config_auto_convertion_verl.jsonc`, whether a mapping to this config already exists.

3. if not, add a config under `ajet` field in `codebase/agentjet/ajet/default_config/ajet_default.yaml`, and add a mapping in `codebase/agentjet/ajet/default_config/verl/config_auto_convertion_verl.jsonc`

4. double check, confirm that default value in `ajet_default.yaml` is the same as verl config in `verl_default.yaml`, and the mapping is correct in `config_auto_convertion_verl.jsonc`

5. ask user whether to add to AgentJetJob (ajet/copilot/job.py), if the user confirms:
  - learn how other config is added in ajet/copilot/job.py
  - add to __init__ signature (with type hint and default None)
  - update docstring with parameter description
  - add instance attribute assignment with cast()
  - add mapping to `overrides` dict

6. **CRITICAL**: update `ajet/default_config/ajet_config_schema.py`
  - the schema must have a dataclass for EVERY nested level in the config path
  - e.g., for `ajet.trainer_common.optim.lr`, need:
    - `AjetOptim` dataclass with `lr: float = 1e-6`
    - `AjetTrainerCommon` must have `optim: AjetOptim = field(default_factory=AjetOptim)`
  - if parent dataclass is missing the nested field, config loading will store it as a raw dict instead of a typed dataclass, causing `getattr()` to fail at runtime
