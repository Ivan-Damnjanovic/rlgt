# Reinforcement Learning for Graph Theory (RLGT)

**Reinforcement Learning for Graph Theory (RLGT)** is a modular reinforcement learning (RL) framework designed to support research in extremal graph theory. RLGT aims to systematize and extend previous RL-based approaches for constructing extremal graphs and counterexamples to graph-theoretic conjectures. The framework provides a clean, modular and extensible codebase suitable for future research.

## Motivation

Reinforcement learning provides a natural formalism for combinatorial optimization. In this approach, an agent interacts with an environment by iteratively modifying a configuration and receives rewards based on a target objective function. In the context of extremal graph theory, RLGT enables researchers to:

* construct graphs that maximize or minimize a given invariant;
* search for counterexamples to conjectured inequalities; and
* discover structural patterns that suggest new theoretical insights.

RLGT bridges computational efficiency, through vectorized graph operations, with flexibility, by offering multiple environments and RL methods, providing a unified and extensible research tool.

## Design Principles

RLGT is implemented in **Python** and follows a layered, object-oriented design. The framework is organized into three packages:

### 1. `graphs`

The `graphs` package provides a core graph abstraction supporting eight different graph formats. All required conversions between these formats are automatically performed. This package supports:

* undirected and directed graphs;
* graphs with or without loops;
* arbitrarily many edge colors; and
* batches of graphs for efficient vectorized operations using **NumPy**.  

The `graphs` package has no internal dependencies and serves as the foundational layer of the framework.

### 2. `environments`

The `environments` package implements RL environments specialized for graph-theoretic problems. It contains nine environments implemented as seven classes and includes auxiliary utilities for deterministic and nondeterministic graph generation. This package depends only on `graphs` and builds reinforcement learning environments on top of the graph abstractions.

### 3. `agents`

The `agents` package implements RL algorithms using **PyTorch**. The available methods include:

* Deep Cross-Entropy;
* REINFORCE; and
* Proximal Policy Optimization (PPO).

This package depends on `graphs` and `environments`, and requires **PyTorch** to be installed. However, using this package is optional; users may choose to work only with the `graphs` and `environments` packages and provide their own RL methods if desired. The package provides fully encapsulated agent implementations that are decoupled from the environment logic.

## Layered Architecture

* **graphs**  
  This package has no internal dependencies and serves as the foundational layer of the framework.

* **environments**  
  This package depends only on `graphs` and builds reinforcement learning environments on top of the graph abstractions.

* **agents**  
  This package depends on `graphs` and `environments`, and requires **PyTorch**. Using this package is optional, and it provides fully encapsulated implementations of reinforcement learning algorithms using **PyTorch**.

## Repository Structure

```
.
├── src/rlgt/
├── tests/
├── docs/
├── examples/
└── applications/
```

* [`src/rlgt`](./src/rlgt) contains the full modular implementation of the framework.
* [`tests`](./tests) contains the unit tests that enhance the code stability.
* [`docs`](./docs) contains the documentation that provides detailed explanations and usage guidelines.
* [`examples`](./examples) contains several examples that demonstrate how to define graphs and environments, and train agents.  
* [`applications`](./applications) contains the applications of this framework to concrete graph theory problems.

## Tooling and Code Quality

RLGT emphasizes reproducibility, stability and clean code. The framework uses:

* **Poetry** — for dependency management and packaging;
* **Black** — for automatic code formatting;
* **isort** — for consistent import sorting; and  
* **pytest** — for unit testing of framework features.

**Poetry** manages both required and optional dependencies, ensuring a clean and reproducible setup. **Black** and **isort** enforce a consistent code style, and **pytest** guarantees reliability through automated testing.

## Installation

TODO

## Documentation

Detailed documentation is available at [`https://ivan-damnjanovic.github.io/rlgt/`](https://ivan-damnjanovic.github.io/rlgt/).

## Citation

If you use RLGT in academic work, please cite the associated paper:

TODO
