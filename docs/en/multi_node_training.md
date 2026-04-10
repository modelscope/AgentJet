# Multi-Node Multi-GPU Training

AgentJet supports scaling training across multiple machines and GPUs. This guide covers how to set up multi-node training in both **Classic Mode** and **Swarm Mode**.

---

## Prerequisites

Before starting multi-node training, ensure that:

1. All nodes have AgentJet installed and configured (see [Installation](../en/installation.md)).
2. All nodes can communicate with each other over the network.
3. A [Ray](https://docs.ray.io/) cluster is properly set up across all nodes.

### Setting Up the Ray Cluster

You have two options to set up Ray:

=== "Auto Configuration"

    Use the built-in helper to automatically configure Ray based on cluster environment variables:

    ```bash
    ajet --with-ray-cluster
    ```

    This command reads the following environment variables and initializes a Ray cluster automatically:

    | Environment Variable | Description |
    |---|---|
    | `MASTER_ADDR` | The hostname or IP address of the head node. AgentJet compares this with the current node's hostname (`os.uname().nodename`) to determine whether to start a Ray **head** node or a **worker** node. |
    | `MASTER_PORT` | The port used by the Ray head node for cluster communication. |

    **How it works:**

    - If the current node's hostname matches `MASTER_ADDR`, AgentJet starts a **Ray head** node: `ray start --head --node-ip-address=$MASTER_ADDR --port=$MASTER_PORT`
    - Otherwise, AgentJet starts a **Ray worker** node that connects to the head: `ray start --address=$MASTER_ADDR:$MASTER_PORT`

    !!! warning "Cluster Compatibility"
        Currently, `--with-ray-cluster` is designed for clusters that provide `MASTER_ADDR` and `MASTER_PORT` environment variables (e.g., Alibaba PAI DLC). For other cluster schedulers (SLURM, Kubernetes, etc.), you may need to set these environment variables manually or use the manual configuration method below.

=== "Manual Configuration"

    Set up Ray manually by starting a head node and connecting worker nodes:

    ```bash
    # On the head node
    ray start --head --port=6379

    # On each worker node
    ray start --address='<head-node-ip>:6379'
    ```

    Verify the cluster is running:

    ```bash
    ray status
    ```

---

## Classic Mode

In Classic Mode, multi-node training is straightforward. After the Ray cluster is ready, simply update your YAML configuration to specify the number of nodes and GPUs per node.

### Step 1: Configure Ray Cluster

Set up the Ray cluster using either method above.

### Step 2: Update Training Configuration

Modify your YAML config to specify the multi-node topology:

```yaml
ajet:
  trainer_common:
    nnodes: 4          # number of machines
    n_gpus_per_node: 8 # number of GPUs per machine
```

### Step 3: Launch Training

Run training as usual:

```bash
ajet --conf your_config.yaml --backbone='verl'
```

AgentJet will automatically distribute the workload across all 4 nodes (32 GPUs total in this example).

!!! tip "Scaling Tips"
    - Set `nnodes` to the total number of machines in your Ray cluster.
    - Set `n_gpus_per_node` to match the number of GPUs available on each machine.
    - Ensure all nodes have identical GPU configurations for optimal performance.

---

## Swarm Mode

In Swarm Mode, multi-node training allows you to launch a distributed swarm server that spans multiple GPU machines, enabling training of larger models.

### Step 1: Configure Ray Cluster

Set up the Ray cluster across all GPU nodes using either method above.

### Step 2: Start the Swarm Server

Launch the swarm server on the head node:

```bash
ajet-swarm start
```

The swarm server will leverage the entire Ray cluster for model hosting and training.

### Step 3: Submit a Multi-Node Job from Client

From any machine (including a GPU-less laptop), submit a training job with multi-node configuration:

```python
from ajet_swarm import AgentJetJob

ajet_job = AgentJetJob(
    base_yaml_config="tutorial/example_werewolves_swarm/werewolves.yaml",
    # the YAML config should also set nnodes > 1, e.g.:
    #   ajet:
    #     trainer_common:
    #       nnodes: 4
    #       n_gpus_per_node: 8
)
```

The YAML referenced by `base_yaml_config` should contain the same multi-node settings:

```yaml
ajet:
  trainer_common:
    nnodes: 4          # number of machines
    n_gpus_per_node: 8 # number of GPUs per machine
```

!!! info "Swarm Advantage"
    With Swarm Mode, you can submit multi-node training jobs remotely without direct access to the GPU cluster. The swarm server coordinates all distributed training internally.

---

## Summary

| | Classic Mode | Swarm Mode |
|---|---|---|
| **Ray Setup** | Required on all nodes | Required on all GPU nodes |
| **Config** | Set `nnodes` and `n_gpus_per_node` in YAML | Same YAML config, submitted via `AgentJetJob` |
| **Launch** | `ajet --conf ...` on head node | `ajet-swarm start` on head node, submit from client |
| **Remote Submit** | Not supported | Supported (GPU-less laptop) |
