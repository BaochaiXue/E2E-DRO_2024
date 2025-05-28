# Forked: Distributionally Robust End-to-End Portfolio Construction - 2024

Welcome to the forked version of the [Distributionally Robust End-to-End Portfolio Construction](https://arxiv.org/abs/2206.05134) repository. This fork includes modifications to ensure compatibility with **Python 3.12** and **PyTorch 2.x**, enabling improved performance and leveraging the latest features of these platforms.

## Table of Contents

- [Overview](#overview)
- [Original Project](#original-project)
- [Key Enhancements](#key-enhancements)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

## Overview

This repository is a fork of the original [Distributionally Robust End-to-End Portfolio Construction](https://arxiv.org/abs/2206.05134) project. The primary objective of this fork is to update the codebase to be compatible with **Python 3.12** and **PyTorch 2.x**, ensuring that users can leverage the latest advancements and maintain compatibility with modern development environments.

## Original Project

The original repository was developed by [Giorgio Costa](https://gcosta151.github.io) and [Garud Iyengar](http://www.columbia.edu/~gi10/), from the Iyengar Lab in the IEOR Department at Columbia University. The project accompanies their paper:

**[Distributionally Robust End-to-End Portfolio Construction](https://arxiv.org/abs/2206.05134)**

This work introduces a robust end-to-end (E2E) learning system where the final decision layer leverages a distributionally robust (DR) optimization model for portfolio construction. The system integrates asset return prediction with DR optimization, allowing for the learning of risk-tolerance parameters and the degree of robustness directly from data.

## Key Enhancements

This fork introduces the following key enhancements to the original repository:

- **Python 3.12 Compatibility**: Updated the codebase to leverage new features and optimizations available in Python 3.12.
- **PyTorch 2.x Support**: Upgraded PyTorch dependencies to version 2.x, ensuring compatibility with the latest PyTorch features and improvements.
- **Dependency Updates**: Refreshed and optimized other dependencies to align with Python 3.12 and PyTorch 2.x requirements.

## Dependencies

Ensure you have the following dependencies installed. Versions have been updated to support Python 3.12 and PyTorch 2.x:

- **Python 3.12**
- **NumPy**: `>=1.23.0`
- **SciPy**: `>=1.9.0`
- **Pandas**: `>=1.5.0`
- **Matplotlib**: `>=3.6.0`
- **cvxpy**: `>=1.3.0`
- **cvxpylayers**: `>=0.1.7`
- **diffcp**: `>=1.0.15` *(required only for models that use the differentiable optimization layers)*
- **PyTorch**: `>=2.0.0`
- **pandas_datareader**: `>=0.10.0`
- **yfinance**: `>=0.2.0`

## Installation

Create a virtual environment and install the dependencies listed in
`requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Example

Experiments are controlled via a YAML configuration file. After installing the
dependencies you can execute them with:

```bash
python examples/main.py
```

The default `examples/config.yaml` enables all experiments (`exp1`-`exp5`).
Edit this file to run a subset.

The scripts cache downloaded data and intermediate results in the
`cache/exp/` directory (and `cache/exp5/` for the synthetic-data example).
These folders are created automatically when running the loader so no manual
setup is required.

## Experiment Descriptions

The `examples/config.yaml` file lists five experiments. Each corresponds to a
different training scenario:

- **exp1** – *General evaluation*: evaluates equal-weight, predict‑then‑optimize,
  base, nominal and distributionally robust (DR) systems on historical data.
  All parameters are learnable.
- **exp2** – *Learning delta*: keeps the prediction weights and risk appetite
  fixed while learning the DR robustness size \(\delta\) to gauge its impact.
- **exp3** – *Learning gamma*: fixes the prediction weights but learns the
  risk‑aversion parameter \(\gamma\), optionally together with \(\delta\).
- **exp4** – *Learning \(\theta\)*: focuses on updating the prediction layer
  while \(\gamma\) (and \(\delta\)) remain fixed, comparing base, nominal and DR
  models.
- **exp5** – *Synthetic data*: applies nominal and DR systems to synthetic
  returns using linear, two‑layer and three‑layer networks.

Edit the YAML file to enable or disable specific experiments as needed.

## Testing

Basic tests are provided using `pytest`:

```bash
pytest
```


## Acknowledgments

- Original repository by [Giorgio Costa](https://gcosta151.github.io) and [Garud Iyengar](http://www.columbia.edu/~gi10/), Iyengar Lab, IEOR Department, Columbia University.
- Thanks to the contributors and the open-source community for their valuable tools and libraries.

