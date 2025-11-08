# Helix — Operator‑Algebraic Diagnostics for Deep Nets

![[IMG_0444.jpeg]]

## What Problem Does Helix Solve?

**The Challenge:** Deep neural networks are black boxes. We train them, but we don't understand *how* they organize information internally or *why* they sometimes fail.

**The Insight:** Neural networks can be viewed as geometric transformers—each layer reshapes the input space. ReLU networks partition space into polyhedral regions, nonlinear layers mix information, and residual connections create flows. These are mathematical structures we can measure.

**The Solution:** Helix provides diagnostic tools that expose these internal structures through stable mathematical invariants, giving you interpretable readouts about what your network is actually doing.

## What Helix Measures (and Why It Matters)

### 1. **Region Partitions** — How your network divides space

**Logic:** ReLU activations split input space into polyhedral regions. Each layer refines these regions further.

**What Helix extracts:**
- **Incidence matrices B_k** — Which parent regions split into which child regions
- **Mass distributions τ_k** — How much data falls into each region
- **Region counts** — Complexity indicator (too few = underutilized, too many = overfitting risk)

**Why it matters:** Reveals if your network is using its capacity effectively or wasting neurons.

### 2. **Information Flow** — How data transforms between layers

**Logic:** Each layer acts as an operator on data. Mathematically, these are completely positive (CP) maps that preserve positivity and probability structure.

**What Helix computes:**
- **CP map Φ(X) = V* X V** — The transformation operator from incidence and masses
- **Unitality check** — Does the layer preserve total probability? (Φ(I) ≈ I)
- **Coisometry check** — Is information preserved or lost? (V V* ≈ I)

**Why it matters:** Detects information bottlenecks, gradient flow issues, and layer health.

### 3. **Mixing Dynamics** — How residual/invertible blocks stir information

**Logic:** Residual connections and invertible layers act like dynamical systems. The Ulam–Perron–Frobenius operator measures how quickly information mixes.

**What Helix measures:**
- **Spectral gap (1 - |λ₂|)** — Mixing speed (larger = faster convergence)
- **Transfer operator P** — Discretized flow dynamics

**Why it matters:** Predicts training stability and convergence properties.

### 4. **Topology Signals** — Shape of your data manifold

**What Helix computes:**
- **Betti numbers (β₀, β₁, β₂)** — Connected components, loops, voids
- **Persistent homology** — Which topological features are robust vs. noise

**Why it matters:** Ensures your network architecture matches your data's geometric structure.

### 5. **Capacity Health** — Are your layers learning or dying?

**What Helix monitors:**
- **Singular value spectra** — Effective dimensionality of each layer
- **Capacity loss scores** — Flags plasticity collapse (dead neurons)

**Why it matters:** Early warning system for training pathologies.

## Installation

**Requirements:**

- Python 3.10+
- Virtual environment (recommended)

**Setup:**

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
# Optional extras
python3 -m pip install torch sympy matplotlib ripser
```

## How to Run Helix

### 1. Interactive Menu (Easiest Start)

Simply run helix without arguments to access the interactive menu:

```bash
./helix
```

This launches a menu with six options:

- **[1] Run Helix environment** — Analyze models in the built-in verification environments
- **[2] Helix environment summary** — View available environments and their configurations
- **[3] Run demo** — Quick demo using the two moons dataset (great first test!)
- **[4] Analyze custom model/data** — Bring your own PyTorch model and data
- **[5] Interactive TUI mode** — Full-featured terminal UI with live visualization
- **[6] Exit** — Quit the program

**Recommended first run:** Choose option `[3]` to see Helix in action on a simple classification task.

### 2. Direct CLI Mode (Scripting & Automation)

Skip the menu and run diagnostics directly with command-line flags:

```bash
./helix \
  --no-train \
  --ulam-bins 25 \
  --ulam-samples-per-cell 4 \
  --plot --no-show \
  --save-prefix helix_out
```

Generates: `helix_out_regions.png`, `helix_out_mass_consistency.png`, `helix_out_cp.png`, `helix_out_ulam.png`.

### 3. Terminal UI Mode (Advanced Interactive)

For live plots and richer interactivity, install TUI extras and launch:

```bash
python3 -m pip install -e .[tui]
./helix tui  # or: helix-tui
```

### 4. Programmatic API (Integration)

Embed Helix diagnostics into your training loops or notebooks:

```python
import numpy as np
import torch, torch.nn as nn
from helix import (
    extract_partitions, build_V_from_incidence, sanity_check_ucp,
    ulam_pf, spectral_gap, region_counts, mass_consistency_errors,
)

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
X = np.random.randn(2000, 2).astype(np.float32)

af = extract_partitions(model, X)
print("regions:", region_counts(af.B_list))
print("mass L1:", mass_consistency_errors(af.B_list, af.tau_list))

cp_stats = []
for k, B in enumerate(af.B_list, start=1):
    tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k-2]
    tau_cur = af.tau_list[k-1]
    V = build_V_from_incidence(B, tau_prev, tau_cur)
    cp_stats.append(sanity_check_ucp(V))
print(cp_stats)

F = lambda x: x  # identity toy map
lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
P, _ = ulam_pf(F, (lo, hi), bins_per_dim=20, samples_per_cell=4)
print("spectral gap:", spectral_gap(P))
```

Project Layout
- code/helix: core library modules (partitions, cp, ulam, ktheory, diagnostics, sparse, plotting, TUI/CLI wrappers)
- code/examples: runnable demo
- environments/: verifiers/eval environments (`helixenv`, `bixbench`, `hle`)
- docs/: user and developer documentation (overview, concepts, API, CLI/TUI, troubleshooting)
- code/tests: unit tests (direct invocation supported; see docs)
- helix: local CLI helper to run the package from the repo without installing

Documentation
- Start here: docs/overview.md
- Concepts: docs/concepts.md
- Applications: docs/applications.md
- Getting started: docs/getting-started.md
- CLI: docs/cli.md, TUI: docs/tui.md
- API surface: docs/api.md
- Troubleshooting: docs/troubleshooting.md
- Environments: docs/environments.md
- Roadmap: docs/roadmap.md

MIT License

Copyright (c) 2025 Project Helix
