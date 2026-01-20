# Helix Overview

## What is Helix?

Helix is a diagnostic toolkit that makes neural networks interpretable by measuring their internal geometric and algebraic structure.

## The Core Idea

**Neural networks transform space.** Each layer reshapes the input space in a specific way:

1. **ReLU layers** partition space into polyhedral regions
2. **Dense/conv layers** mix and transform information
3. **Residual connections** create dynamical flows

Helix measures these transformations using stable mathematical invariants from operator algebra and dynamical systems theory.

## What Helix Measures

### 1. **Partition Structure** (AF algebras)

**What:** How ReLU layers divide input space into regions

**How:** Extract incidence matrices B_k (parent→child splits) and mass distributions τ_k (data per region)

**Why:** Reveals capacity utilization, overfitting risk, and architectural efficiency

### 2. **Information Flow** (CP maps)

**What:** How data transforms between layers

**How:** Build completely positive maps Φ(X) = V* X V from geometric data

**Why:** Detects information bottlenecks, gradient flow issues, and layer health via unitality/coisometry checks

### 3. **Mixing Dynamics** (Ulam-Perron-Frobenius)

**What:** How residual/invertible blocks stir information

**How:** Discretize space and compute transfer operator P with spectral gap analysis

**Why:** Predicts training stability and convergence properties

## Why This Approach Works

**Stability:** Mathematical invariants are robust to noise and perturbations

**Scalability:** Sparse data structures (O(n) memory) and efficient algorithms work from toy models to large networks

**Interpretability:** Each diagnostic has clear geometric meaning and actionable insights

## Core Modules

- **`helix.partitions`** — Extract ReLU partitions, incidence matrices, and mass distributions
- **`helix.cp`** — Build CP map operators and verify unitality/coisometry/PSD properties
- **`helix.ulam`** — Compute Ulam transfer operators with barycentric sampling and spectral analysis
- **`helix.ktheory`** — K-theory invariants (Smith normal form) for stationary systems
- **`helix.diagnostics`** — High-level metrics for training-time monitoring
- **`helix.plotting`** — Visualization tools for all diagnostics

## Getting Started

**Quick start:**
1. Run `./helix` and choose option [3] for an interactive demo
2. Read `docs/concepts.md` to understand the mathematical foundations
3. See `docs/getting-started.md` for integration into your workflow

**For researchers:** The mathematical theory is explained in detail in `docs/concepts.md` using logical deduction from first principles.

**For practitioners:** The CLI and TUI provide immediate insights without requiring deep mathematical knowledge.

