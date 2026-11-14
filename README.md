# Tensor-Neural-Net-Repo

## Summary

This repo is for implementing tensor based neural networks primarily using the pytorch module. The modules implemented begin from elementary and move on to more advanced.


## Repo Structure

```bash

Tensor-Neural-Net-Repo/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .pre-commit-config.yaml
│
├── configs/
│   ├── mlp.yaml
│   ├── cnn.yaml
│   ├── rnn.yaml
│   └── transformer.yaml
│
├── src/
│   └── tensor_neural_net/
│       ├── __init__.py
│       ├── cli.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── mlp.py
│       │   ├── cnn.py
│       │   ├── resnet.py
│       │   ├── rnn.py
│       │   ├── lstm.py
│       │   ├── gru.py
│       │   ├── attention.py
│       │   ├── transformer.py
│       │   └── autoencoder.py
│       ├── layers/
│       │   ├── __init__.py
│       │   ├── activations.py
│       │   ├── normalization.py
│       │   ├── dropout.py
│       │   └── positional_encoding.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   ├── losses.py
│       │   ├── optimizers.py
│       │   ├── schedulers.py
│       │   ├── callbacks.py
│       │   └── checkpoint.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py
│       │   ├── preprocessing.py
│       │   └── dataloaders.py
│       └── utils/
│           ├── __init__.py
│           ├── metrics.py
│           ├── visualization.py
│           ├── reproducibility.py
│           └── model_summary.py
├── examples/
│   ├── 01_xor.py
│   ├── 02_house_price.py
│   ├── 03_mnist_cnn.py
│   ├── 04_sequence_prediction.py
│   └── 05_transformer.py
├── experiments/
│   ├── activation_comparison.py
│   ├── optimizer_comparison.py
│   ├── cnn_vs_mlp.py
│   └── rnn_vs_lstm.py
├── docs/
│   ├── neural_network_basics.md
│   ├── backpropagation.md
│   ├── convolution.md
│   ├── recurrent_networks.md
│   ├── attention.md
│   └── optimization.md
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_layers.py
│   ├── test_training.py
│   ├── test_gradients.py
│   └── test_cli.py
└── .github/
    └── workflows/
        └── tests.yml


```
