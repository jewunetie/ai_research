# Experiment Configurations

This directory contains YAML configuration files for all experiments.

## Configuration Files

### MNIST Experiments

| Config | Approach | Description |
|--------|----------|-------------|
| `mnist_baseline_bp.yaml` | Pure BP | Standard backprop baseline |
| `mnist_baseline_ff.yaml` | Pure FF | Forward-Forward baseline |
| `mnist_sequential_phased.yaml` | Hybrid | 3-phase: FF pretrain → BP classifier (frozen) → Fine-tune all |
| `mnist_detached_interface.yaml` | Hybrid | Simultaneous FF+BP with gradient detachment |

### Fashion-MNIST Experiments

| Config | Approach | Description |
|--------|----------|-------------|
| `fashion_mnist_baseline_bp.yaml` | Pure BP | Standard backprop baseline |
| `fashion_mnist_baseline_ff.yaml` | Pure FF | Forward-Forward baseline |
| `fashion_mnist_sequential_phased.yaml` | Hybrid | 3-phase hybrid approach |

### CIFAR-10 Experiments

| Config | Approach | Description |
|--------|----------|-------------|
| `cifar10_baseline_bp.yaml` | Pure BP | Standard backprop baseline on color images |
| `cifar10_sequential_phased.yaml` | Hybrid | 3-phase hybrid approach on color images |

## Running Experiments

### Single Experiment

```bash
python experiments/run_experiment.py --config experiments/configs/mnist_baseline_bp.yaml
```

### All Baselines for a Dataset

```bash
# Run all MNIST baselines
python experiments/run_baselines.py --dataset mnist

# Run all Fashion-MNIST baselines
python experiments/run_baselines.py --dataset fashion_mnist

# Run all CIFAR-10 baselines
python experiments/run_baselines.py --dataset cifar10
```

### All Experiments

```bash
python experiments/run_baselines.py --all
```

## Configuration Structure

Each config file has the following structure:

```yaml
experiment:
  name: "experiment_name"
  approach: "pure_backprop|pure_ff|sequential_phased|detached_interface"
  seed: 42
  device: "auto"  # auto, mps, cuda, cpu

data:
  dataset: "mnist|fashion_mnist|cifar10"
  batch_size: 512
  num_workers: 4
  normalize: true

model:
  architecture: "mlp|ff_network|hybrid_ff_bp"
  # Architecture-specific parameters...

training:
  # Approach-specific training parameters...

evaluation:
  save_checkpoints: true
  checkpoint_freq: 20
  log_interval: 10

logging:
  use_tensorboard: true
  save_dir: "results/experiment_name"
```

## Creating Custom Configs

1. Copy an existing config as a template
2. Modify the experiment name and parameters
3. Save with a descriptive name
4. Run with `run_experiment.py --config <your_config>.yaml`

## Approach-Specific Parameters

### Pure Backpropagation (`pure_backprop`)

```yaml
model:
  architecture: "mlp"
  hidden_dims: [500, 500, 500]

training:
  num_epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
```

### Pure Forward-Forward (`pure_ff`)

```yaml
model:
  architecture: "ff_network"
  hidden_dims: [500, 500, 500]

training:
  num_epochs: 100
  learning_rate: 0.03
  threshold: 2.0
  negative_strategy: "random_label"
  layer_wise: true
```

### Sequential Phased (`sequential_phased`)

```yaml
model:
  architecture: "hybrid_ff_bp"
  ff_hidden_dims: [500, 500, 500]

training:
  approach: "sequential_phased"

  phase1:  # FF Pretraining
    epochs: 100
    learning_rate: 0.03
    threshold: 2.0
    negative_strategy: "random_label"

  phase2a:  # BP Classifier (FF frozen)
    epochs: 50
    learning_rate: 0.001
    freeze_ff: true

  phase2b:  # Fine-tune all
    enabled: true
    epochs: 50
    learning_rate: 0.0001
```

### Detached Interface (`detached_interface`)

```yaml
model:
  architecture: "hybrid_ff_bp"
  ff_hidden_dims: [500, 500, 500]

training:
  num_epochs: 150

  ff_config:
    learning_rate: 0.03
    threshold: 2.0
    negative_strategy: "random_label"

  bp_config:
    learning_rate: 0.001
    optimizer: "adam"
```
