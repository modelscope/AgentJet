# Frozen Lake

This example shows the usage of GRPO on the [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) task.


## Data and Environment Preparation

After setting up the basic environment following the [installation guidance](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_installation.html), you need to install the additional dependencies by running the following command:

```bash
pip install gymnasium[toy_text]
```

## begin training

python launcher.py --conf tutorial/example_frozenlake/frozenlake.yaml --backbone=verl
