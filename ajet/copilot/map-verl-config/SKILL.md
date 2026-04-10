---
name: map-verl-config
description: map verl config to agentjet config
license: Complete terms in LICENSE.txt
---


1. find user requested verl config in in codebase/agentjet/ajet/default_config/verl/verl_default.yaml

2. check `codebase/agentjet/ajet/default_config/verl/config_auto_convertion_verl.jsonc`, whether a mapping to this config already exists.

3. if not, add a config under `ajet` field in `codebase/agentjet/ajet/default_config/ajet_default.yaml`, and add a mapping in `codebase/agentjet/ajet/default_config/verl/config_auto_convertion_verl.jsonc`

4. double check, confirm that default value in `ajet_default.yaml` is the same as verl config in `verl_default.yaml`, and the mapping is correct in `config_auto_convertion_verl.jsonc`