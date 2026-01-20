# Getting Started with Helix

This guide walks you through your first Helix analysis in 5 minutes.

## Step 1: Installation

**Create a virtual environment (recommended):**

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip
```

**Install Helix:**

```bash
python3 -m pip install -e .
```

**Install optional dependencies:**

```bash
# For neural network analysis
python3 -m pip install torch

# For visualization
python3 -m pip install matplotlib

# For topology analysis
python3 -m pip install ripser

# For symbolic computation
python3 -m pip install sympy
```

## Step 2: Run Your First Demo

**Launch the interactive menu:**

```bash
./helix
```

**Choose option [3]** — This runs a quick demo on the two moons classification dataset.

**What you'll see:**
- Region counts showing how the network partitions space
- Mass consistency errors (should be near zero for healthy networks)
- CP diagnostics (unitality, coisometry, PSD checks)
- Ulam spectral gap (mixing dynamics)

**Interpretation guide:**
- **High region counts** → Network is using its capacity
- **Low mass errors** → Probability is conserved correctly
- **Small unitality error** → Layer preserves information
- **Large spectral gap** → Fast mixing, stable training

## Step 3: Analyze Your Own Model

**Option A: Interactive mode**

```bash
./helix
# Choose option [4] and follow prompts
```

**Option B: Programmatic API**

```python
import numpy as np
import torch.nn as nn
from helix import extract_partitions, region_counts, mass_consistency_errors

# Your model
model = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(),
    nn.Linear(16, 8), nn.ReLU(),
    nn.Linear(8, 2)
)

# Your data
X = np.random.randn(1000, 2).astype(np.float32)

# Extract diagnostics
af = extract_partitions(model, X)
print("Region counts per layer:", region_counts(af.B_list))
print("Mass consistency errors:", mass_consistency_errors(af.B_list, af.tau_list))
```

**See `docs/api.md` for the complete API reference.**

## Step 4: Visualize Results

**Generate plots:**

```bash
./helix --plot --save-prefix my_analysis
```

**Output files:**
- `my_analysis_regions.png` — Region growth across layers
- `my_analysis_mass_consistency.png` — Probability conservation
- `my_analysis_cp.png` — Information flow health
- `my_analysis_ulam.png` — Mixing dynamics spectrum

## Step 5: Advanced Usage

**Interactive TUI with live visualization:**

```bash
python3 -m pip install -e .[tui]
./helix tui
```

**Custom analysis with flags:**

```bash
./helix \
  --samples 5000 \
  --noise 0.05 \
  --width 32 \
  --epochs 150 \
  --ulam-bins 30 \
  --plot
```

## Understanding the Output

### AF Partition Metrics

- **Region counts (n_k):** Number of polyhedral regions per layer
  - Too few → Underutilized capacity
  - Too many → Potential overfitting
  
- **Mass consistency (‖τ_{k-1} − B_k τ_k‖₁):** Probability conservation error
  - Should be < 1e-6 for healthy networks
  - Large errors indicate numerical issues

### CP Map Diagnostics

- **Unitality (‖Φ(I) − I‖_F):** Does the layer preserve total probability?
  - Should be < 0.01 for well-behaved layers
  
- **Coisometry (‖V V* − I‖_F):** Is information preserved?
  - Small values → Reversible transformation
  - Large values → Information bottleneck
  
- **PSD violation:** Minimum eigenvalue of positive matrices
  - Should be ≥ 0 (negative indicates numerical instability)

### Ulam Transfer Operator

- **Spectral gap (1 − |λ₂|):** Mixing speed
  - Close to 1 → Fast mixing, stable dynamics
  - Close to 0 → Slow mixing, potential instability
  
- **Top eigenvalues:** Dominant modes of the flow

## Next Steps

- **Understand the theory:** Read `docs/concepts.md` for mathematical foundations
- **Explore environments:** See `docs/environments.md` for verification benchmarks
- **Integration guide:** Check `docs/cli.md` and `docs/tui.md` for advanced features
- **Troubleshooting:** Visit `docs/troubleshooting.md` if you encounter issues

