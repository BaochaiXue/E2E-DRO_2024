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
- **diffcp**: `>=1.0.15`
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

An example experiment script is available in `examples/main.py`. After
installing the dependencies you can execute it with:

```bash
python examples/main.py
```

The script caches downloaded data and intermediate results in the `cache/`
directory. If the folder does not already exist it will be created
automatically when running the loader.

## Testing

Basic tests are provided using `pytest`:

```bash
pytest
```


## Acknowledgments

- Original repository by [Giorgio Costa](https://gcosta151.github.io) and [Garud Iyengar](http://www.columbia.edu/~gi10/), Iyengar Lab, IEOR Department, Columbia University.
- Thanks to the contributors and the open-source community for their valuable tools and libraries.

