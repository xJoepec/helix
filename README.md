# Helix CLI

## Operator-Algebraic Diagnostics for Deep Learning

> Helix is a diagnostic toolkit that treats neural networks as geometric transformers. Instead of staring at loss curves, you can inspect the operator-algebraic invariants that describe how a model partitions, transports, and mixes information. The goal: every metric is tied back to a mathematical principle so you can reason about your network's internal state.

---

## Why Helix?

- **Interpretable invariants.** ReLU networks induce AF (Approximately Finite) algebras; residual blocks behave like flows; CP maps expose information conservation. Helix measures these invariants directly.
- **Actionable metrics.** Region counts, mass consistency, CP unitality/coisometry, Ulam spectral gaps, K-theory signatures, persistent homology, and capacity scores turn abstract math into concrete health checks.
- **Multiple front-ends.** Use an interactive CLI, a Textual-powered TUI, or the Python API to embed diagnostics in training loops, scripts, or notebooks.
- **Research-grade yet production-aware.** Sparse parent-pointer structures keep memory linear, optional dependencies are isolated, and core diagnostics ship with tests and runnable docs.
- **Pedagogical documentation.** `docs/concepts.md` walks from first principles to full diagnostics; CLI/TUI guides, environment references, and troubleshooting notes complete the learning arc.

---

## The Conceptual Ladder

| Step | Mathematical Object | What Helix Extracts | Why it Matters |
| --- | --- | --- | --- |
| **1. AF Partitions** | ReLU refinements of polyhedral regions (Bratteli diagrams) | Incidence matrices `B_k ∈ {0,1}^{n_{k-1}×n_k}`, mass vectors `τ_k` with `τ_{k-1} ≈ B_k τ_k`, region counts | Capacity utilization, inductive bias, and degeneracy detection |
| **2. CP Maps** | Completely positive maps `Φ_k(X) = V_k* X V_k` built from `B_k` and `τ` | Kraus/Stinespring operator `V_k[p,j] = √(τ_k[j]/τ_{k-1}[p])` (with zero-mass guards), unitality/coisometry errors `‖V_k*V_k - I‖_F`, PSD margins | Quantifies information conservation and gradient transport per layer |
| **3. Mixing Dynamics** | Ulam-Perron-Frobenius operator `P` for residual flows | Grid/bin construction, barycentric pushforward, eigenvalue magnitudes, spectral gap `1 - |lambda_2|` | Diagnoses chaotic vs. contractive dynamics, neural ODE stability |
| **4. Topology & K-Theory** | Persistent homology, K0/K1 groups, Smith normal form | Betti numbers (beta0, beta1, beta2), lifetime summaries, torsion coefficients, rank/nullity | Tracks manifold shifts, model provenance, architecture equivalence classes |
| **5. Capacity Health** | Layer weight spectra, combinatorial growth, anisotropy proxy | Capacity loss ratios from singular values, cumulative anisotropy column sums, combinatorial entropy | Early warnings for plasticity collapse, dead neurons, or overconstrained blocks |

Each diagnostic is optional yet composable: start with AF partitions, layer in CP checks, then escalate to topology or spectral dynamics when the experiment demands it.

### Operator Diagnostics at a Glance

- **Mass recursion:** `τ_{k-1} = B_k τ_k + δ_k` with `‖δ_k‖₁` reported as mass inconsistency. Parent-pointer storage can keep this O(n) memory.
- **CP sanity:** Helix reports `unital_err_fro`, `coisometry_err_fro`, and `psd_min_eig_violation` from CP sanity checks.
- **Spectral mixing:** Helix computes spectral gaps from the Ulam PF operator; the CLI/demo uses a small synthetic residual block unless you provide your own map via `ulam_pf`.
- **Topology/K-theory:** Smith normal form of `I - B_k^T` reveals torsion and rank/nullity; persistent homology returns Betti numbers and lifetime summaries from point clouds.
- **Capacity signals:** Capacity loss uses layer weight singular values; anisotropy proxy uses cumulative incidence column sums.

---

## Installation & Requirements

Helix targets Python 3.10+. A virtual environment is strongly recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip

# Choose the feature set that matches your workflow:
python3 -m pip install -e .                # Minimal API (partition extraction only)
python3 -m pip install -e .[torch,viz]     # CLI + plotting (PyTorch + matplotlib)
python3 -m pip install -e .[tui]           # TUI interface (textual, rich)
python3 -m pip install -e .[full]          # Everything: torch, viz, tui, topology
```

> **PyTorch is required** for the demo, CLI analysis, and most environments. Install `torch` separately if you pick the minimal target.

---

## Run Helix Three Ways

### Keyless Unsloth monitoring

Observe a local Unsloth Studio run without loading a second model or mutating Studio:

```bash
helix grok-watch --studio-url http://127.0.0.1:8888 --once --json
```

See [`docs/grok-watch.md`](docs/grok-watch.md) for the loopback and GPU-safety contract.

### 1. Interactive Menu (on-ramp)

```bash
./helix
```

Options include:

1. Run built-in Helix environments (verification-ready setups)
2. Summarize available environments and their parameters
3. Demo on the classic two moons dataset
4. Analyze your own PyTorch model/data
5. Launch the TUI from the menu
6. Exit

> Tip: Option `[3]` demonstrates every diagnostic end-to-end in under a minute.

### 2. Direct CLI (automation)

```bash
./helix \
  --no-train \
  --ulam-bins 25 \
  --ulam-samples-per-cell 4 \
  --plot --no-show \
  --save-prefix helix_out
```

Outputs:

- `helix_out_regions.png` — partition growth
- `helix_out_mass_consistency.png` — τ consistency
- `helix_out_cp.png` — unitality/coisometry traces
- `helix_out_ulam.png` — Ulam eigenvalue spectrum (|lambda|)

CLI niceties:

- `--animate-forward` renders a colorized activation sweep before summaries (TTY only).
- Automatic `torchinfo`/`torchsummary` dumps (disable via `HELIX_SKIP_MODEL_SUMMARY=1`).

### 3. Textual UI (live exploration)

```bash
python3 -m pip install -e .[tui]
helix tui   # or: helix-tui
```

Features:

- Background computation with cancel and rerun
- Persistent configuration (`~/.config/helix/config.toml`)
- Exportable JSON/CSV snapshots (`helix_metrics_YYYYMMDD_HHMMSS.json`)
- Keyboard shortcuts: `r` (run), `c` (cancel), `s` (save), `d` (reset), `e` (export JSON), `x` (export CSV), `:` (command), `h` (help), `q` (quit)
- Live panels for region counts, mass errors, CP diagnostics, Ulam spectral gap

### Programmatic API (embed in code)

```python
import numpy as np, torch.nn as nn
from helix import (
    extract_partitions, region_counts, mass_consistency_errors,
    build_V_from_incidence, sanity_check_ucp,
    ulam_pf, spectral_gap,
)

model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
X = np.random.randn(2000, 2).astype(np.float32)

af = extract_partitions(model, X)
print("regions:", region_counts(af.B_list))
print("mass L1:", mass_consistency_errors(af.B_list, af.tau_list))

tau_prev = np.array([1.0])
for B, tau_cur in zip(af.B_list, af.tau_list):
    V = build_V_from_incidence(B, tau_prev, tau_cur)
    print(sanity_check_ucp(V))
    tau_prev = tau_cur

F = lambda x: x
lo, hi = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
P, _ = ulam_pf(F, (lo, hi), bins_per_dim=20, samples_per_cell=4)
print("spectral gap:", spectral_gap(P))
```

Complete API reference: `docs/api.md`.

---

## Interpreting the Readouts

| Diagnostic | Healthy Signal | Red Flag |
| --- | --- | --- |
| **Region counts `n_k`** | Smooth growth that matches architecture depth | Exploding regions (overfitting) or flat counts (unused neurons) |
| **Mass consistency `‖τ_{k-1} - B_k τ_k‖₁`** | < 1e-6 | Positive probability leak or numerical instability |
| **Unitality `‖Φ(I) - I‖_F`** | < 1e-2 | Layer amplifying/attenuating information unnaturally |
| **Coisometry `‖V V* - I‖_F`** | Small and flat | Information bottlenecks, dead channels |
| **PSD min eigenvalue** | ≥ 0 | Negative => CP map violated (bug or unstable layer) |
| **Spectral gap `1 - |λ₂|`** | Large gap (fast mixing) | Gap → 0 means slow convergence, looming instability |
| **Betti numbers / lifetimes** | Stable across checkpoints | Sudden jumps = topology shift, distribution drift |
| **Capacity scores** | Wide plateau | Drop-off warns of plasticity collapse |

Head to `docs/getting-started.md` for a narrated walk-through of these interpretations.

---

## Use Cases That Benefit from Helix

Helix's operator-algebraic view generalizes beyond "why did my classifier fail?" Here are a few highlights drawn from `uses.md`:

1. **Model provenance.** K-theory invariants (K₀/K₁ via Smith normal forms of `I - Bᵀ` from `helix.ktheory`) act as fingerprints for detecting derived/fine-tuned models.
2. **Phase transition alerts in LLMs.** Track Ulam spectral gaps across checkpoints (optionally with tensor-train Ulam) to spot emergent capability jumps before they appear in evals.
3. **Quantum-inspired compression.** Navigate Morita-equivalent AF algebras to identify thinner networks that preserve computational structure; exploit sparse partition stats to guide pruning.
4. **Market microstructure discovery.** Partition extraction plus τ consistency and anisotropy metrics uncovers changing liquidity regimes in trading models.
5. **Adversarial robustness certification.** Coisometry errors bound perturbation amplification; small ‖V V* − I‖_F becomes a certifiable robustness criterion for residual stacks.

`uses.md` lists even more speculative directions—from protein folding flows to consensus dynamics in swarm robotics and even consciousness studies.

---

## Verification Environments & Benchmarks

- `environments/helixenv` — Lightweight Verifiers-style benchmark validating AF partition metrics.
- `environments/ktheory` — Deep-dive K-theory and persistent homology exploration with scriptable entry points.

Each environment is a real workload that exercises `env_api.py`, so public APIs must stay backward compatible.

---

## Documentation Roadmap

- **Orientation:** `docs/overview.md`
- **Mathematical foundations:** `docs/concepts.md`
- **Workflow guide:** `docs/getting-started.md`
- **CLI/TUI references:** `docs/cli.md`, `docs/tui.md`
- **Applications & recipes:** `docs/applications.md`, `uses.md`
- **Troubleshooting:** `docs/troubleshooting.md`
- **Roadmap:** `docs/roadmap.md`, `docs/roadmap_tui.md`
- **API surface:** `docs/api.md`

Treat the README as the map; the docs are the textbook.

---

## Numerical Backbone

- **Sparse partitions.** `helix.partitions` records parent pointers alongside dense incidence matrices; `helix.sparse` can rebuild B from parents when you want to keep only the sparse representation.
- **Stable CP construction.** `helix.cp` builds `V` directly from masses and incidence, skips zero-mass entries, and reports unitality/coisometry/PSD violations.
- **High-dimensional flows.** `helix.ulam_tensor` uses tensor-train factorization (`O(d·bins²·r²)` instead of `O(bins^d)`) so Ulam/PF analysis stays tractable for d ≥ 3.
- **Topology fallbacks.** `helix.topology` prefers `ripser`; otherwise it returns connectivity-based summaries with clear backend notes.
- **Environment API.** `code/helix/env_api.py` defines the AFMetrics schema for `helixenv`, `ktheory`, and third-party integrations. Keep it backward compatible.

---

## Repository Layout

- `code/helix/` — Core library (partitions, CP, standard and tensor-train Ulam, K-theory, topology, diagnostics, plotting, CLI/TUI entry points, architecture adapters, sparse helpers).
- `code/examples/` — Runnable demos (`helix_demo.py`, tensor-train workflows).
- `code/tests/` — Unit tests for every diagnostic (run with `pytest code/tests -v`).
- `environments/` — External evaluation harnesses (`helixenv`, `ktheory`).
- `ascii-animations/` — ANSI frames for CLI banners.
- `docs/` — Complete documentation set described above.
- `./helix` — Convenience launcher so you can run the CLI directly from the repo.

---

## Development & Quality Checklist

- Keep numerical stability front and center (handle zero masses, guard divisions).
- Optional dependencies: wrap imports with fallbacks (`torch`, `textual`, `ripser`, `sympy`, etc.).
- Before sending changes:

  ```bash
  ruff check .
  pytest -q
  ```

- Never break `env_api.py`; external agents depend on it.
- See `CLAUDE.md` for a detailed assistant/developer playbook.

---

## Troubleshooting & Support

Common issues (missing PyTorch, visualization backends, persistent homology extras, uv vs. pip workflows) are cataloged in `docs/troubleshooting.md`. If something is still unclear:

1. Re-run installation commands inside a fresh virtual environment.
2. Verify extras: `pip list | grep -E "helix|torch|textual"`.
3. File an issue on GitHub with logs and command invocations (`./helix --help | head -1` shows the version).

---

## License

MIT License — Copyright (c) 2025 Project Helix.
