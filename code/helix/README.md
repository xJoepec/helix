## Helix — Operator‑Algebraic Diagnostics for Deep Nets

Helix turns the “deep nets as manifold sculptors” picture into computable diagnostics using operator‑algebraic tools. It exposes stable invariants and practical readouts that scale from toy MLPs to modern architectures.

### What Helix gives you

- **AF partitions (ReLU)**: Extract refining partitions across depth, build incidence `B_k`, masses `τ_k`, and dimension‑group style summaries.
- **CP embeddings (nonlinearity)**: Construct numerically stable unital CP maps `Φ(X)=V* X V` from incidence and masses; sanity‑check unitality/PSD and coisometry.
- **Flow diagnostics (invertible blocks)**: Ulam–Perron–Frobenius discretization with barycentric mass splitting and multi‑samples per cell; read spectral gaps.
- **Sparse structure**: Parent‑pointer representation of refinements for O(n) memory and fast mass aggregation.
- **CLI and plots**: One‑command run to generate region, mass‑consistency, CP, and Ulam spectra plots.

### Install

Use a virtual environment (recommended). From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
# Optional extras
python3 -m pip install torch sympy matplotlib
```

You can also run without installing using the repo script or module entrypoint:

```bash
./helix --help
python -m helix --help
```

### Quickstart (CLI)

```bash
./helix \
  --no-train \
  --ulam-bins 25 \
  --ulam-samples-per-cell 4 \
  --plot --no-show \
  --save-prefix helix_out
```

Generates:
- `helix_out_regions.png`: region counts by depth
- `helix_out_mass_consistency.png`: L1 errors of `τ_{k-1} - B_k τ_k`
- `helix_out_cp.png`: CP diagnostics (unitality, coisometry, PSD)
- `helix_out_ulam.png`: top magnitudes of Ulam PF eigenvalues

### Programmatic API (Python)

Minimal end‑to‑end example:

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

# Ulam PF with barycentric splitting
F = lambda x: x  # identity toy map
lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
P, _ = ulam_pf(F, (lo, hi), bins_per_dim=20, samples_per_cell=4)
print("spectral gap:", spectral_gap(P))
```

### Core concepts (pedagogical)

- **AF partitions and ordered `K₀`**: ReLU nets refine polyhedral partitions with depth. Finite snapshots form a Bratteli diagram (incidence `B_k`). Their inductive limit is AF; the ordered `K₀` colimit of `B_k^T` tracks stable refinement statistics and mass transport across depths.
- **CP maps and Stinespring**: Nonlinear/averaging operations are completely positive on observables. Helix constructs a concrete unital CP map from refinement + masses via `V_k = D(τ_{k-1})^{-1/2} B_k D(τ_k)^{1/2}` (implemented stably, with masking for zero masses) and checks unitality/PSD and coisometry proxies.
- **Flows and crossed products**: Invertible/residual blocks act as *‑automorphisms on observables. Ulam discretization approximates the Perron–Frobenius operator; the second eigenvalue magnitude `|λ₂|` is a practical proxy for mixing, aligning with crossed‑product spectral intuition.
- **Stabilization (width) and Morita**: Channel amplification corresponds to matrix stabilization; invariants like `K`‑theory and traces remain stable under many width/basis changes, explaining why the diagnostics are robust to architecture tweaks.

### Data structures and performance

- **Sparse parents**: Use `parent_of_list` (length `n_k` at depth `k`) instead of dense `B_k`. Convert on demand with `helix.B_from_parents(...)`. Memory becomes O(n) and mass aggregation is O(n) via indexed adds.
- **AFExtraction**: `B_list`, `tau_list`, `parent_of_list`, and `parts` per depth.

### CLI options (selected)

- `--samples`, `--noise`, `--width`, `--epochs`, `--no-train`
- `--ulam-bins`: grid resolution per dimension
- `--ulam-samples-per-cell`: uniform jitter samples per cell for barycentric splitting
- `--plot`, `--no-show`, `--save-prefix`

### Troubleshooting

- “Module not found: helix”: run via `./helix` or `python -m helix`, or install in a venv.
- Matplotlib missing: `python3 -m pip install matplotlib`.
- System Python denies editable installs: use a virtualenv or `./helix` script.

### Testing and style

```bash
python3 code/tests/test_sparse.py -v
ruff check .
```

### References and deeper reading

- See `reference/` for the full paper draft and notes connecting the AF/CP/flow/groupoid machinery to modern ML practice.

### License

MIT

## Project Helix

Computational tools that bridge operator algebras with practical ML diagnostics.

### Install

- Python 3.10+
- Dependencies: numpy, (optional) sympy, pytorch

Using pip:

```
pip install numpy sympy torch
```

### Layout

- `overview.md`: high-level thesis and workflow
- `code/helix`: core library
  - `partitions.py`: ReLU partitions, incidence `B_k`, masses `τ_k`
  - `cp.py`: CP embeddings `Φ(X)=V^* X V` and checks
  - `ulam.py`: Ulam–Perron–Frobenius discretization and spectral gap
  - `ktheory.py`: Smith normal form and toy K-invariants
  - `diagnostics.py`: training-time readouts (growth, anisotropy, mass checks)
- `code/examples/helix_demo.py`: end-to-end demo on a toy MLP
- `helix-latex`: LaTeX paper with proofs and references

### Quickstart

Run the demo:

```
python code/examples/helix_demo.py
```

It prints region counts, mass consistency errors, CP unitality/PSD checks, a flow spectral gap, and an anisotropy proxy.

### Program in one paragraph

ReLU nets induce refining partitions; finite snapshots assemble into an AF inductive system with `K₀` and traces recording refinement and mass transport. Invertible (flow-like) blocks act by automorphisms with crossed-product diagnostics (spectral gap via Ulam). Nonlinear/channel mixing is modeled via CP maps and Stinespring dilations on stabilized algebras, aligning with width/basis changes via Morita equivalence. The provided numerics make these invariants observable on real models.

### Notes

- SymPy is optional (for Smith normal form)
- GPU is not required; CPU suffices for the demo

Alternatively, use the CLI entry script with Ulam sampling control:

```
./helix --plot --ulam-bins 25 --ulam-samples-per-cell 4 --save-prefix helix_out
```

