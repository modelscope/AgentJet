# Trinity Config

## To change trinity config

- please edit `astune/default_config/trinity/trinity_default.yaml`

- or, you can write in your experiment config:

```yaml
trinity:
  algorithm:
    algorithm_type: multi_step_grpo
```

- never edit `astune/default_config/trinity/trinity_default.yaml`



## change config mapping

Some astune config has several configuration that is overlaps with trinity,
map them with `astune/default_config/trinity/config_auto_convertion_trinity.json`