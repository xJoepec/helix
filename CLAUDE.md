# CLAUDE.md — AI Assistant Guide for Helix

This document helps AI assistants understand Helix's architecture, philosophy, and development practices.

## Project Philosophy: Interpretability Through Mathematics

**Core Insight:** Neural networks are geometric transformers. Each layer reshapes space in measurable ways.

**Helix's Approach:** Apply operator algebra and dynamical systems theory to extract stable, interpretable invariants from these transformations.

**Design Principles:**
1. **Pedagogical first** — Code and docs teach the math, not just implement it
2. **Numerically stable** — Handle edge cases (zero masses, singular matrices) gracefully
3. **Production ready** — Sparse representations, efficient algorithms, comprehensive testing
4. **Progressive disclosure** — Simple API for practitioners, deep theory for researchers

## What Helix Does (Logical Chain)

### 1. ReLU Networks → Space Partitions → AF Algebras

**Observation:** ReLU activation `max(0, x)` creates binary decision boundaries
**Deduction:** Layer with n neurons → 2^n potential polyhedral regions
**Key insight:** Deeper layers refine (split) existing regions, never merge
**Mathematical structure:** This refinement sequence is an AF (Approximately Finite) inductive system
**What we track:** Incidence matrices B_k (parent→child splits), masses τ_k (data distribution)
**Implementation:** `code/helix/partitions.py`

### 2. Layer Transformations → CP Maps → Information Flow

**Observation:** Layers transform data while preserving positivity and normalization
**Mathematical fact:** These are completely positive (CP) maps
**Stinespring form:** Any CP map can be written as Φ(X) = V* X V
**What we build:** Operator V from geometric data (B_k, τ_k)
**Health checks:** Unitality (probability conservation), coisometry (information preservation)
**Implementation:** `code/helix/cp.py`

### 3. Residual Blocks → Dynamical Flows → Mixing Analysis

**Observation:** Residual connections x → x + f(x) create flows in activation space
**Question:** How quickly does information mix?
**Method:** Ulam discretization of Perron-Frobenius operator
**Key metric:** Spectral gap = 1 - |λ₂| (convergence rate)
**Implementation:** `code/helix/ulam.py`, `code/helix/ulam_tensor.py` (high-dimensional)

## Documentation Structure (Start Here)

**For first-time users:**
- `README.md` — Problem statement, what Helix measures, why it matters
- `docs/getting-started.md` — 5-minute walkthrough with interpretation guide
- `docs/overview.md` — High-level capabilities and module descriptions

**For understanding the math:**
- `docs/concepts.md` — Mathematical foundations via logical deduction
- Each concept: Basic observation → Deduction → Mathematical structure → Practical use

**For implementation:**
- `docs/api.md` — Python API reference
- `docs/cli.md` — Command-line interface
- `docs/tui.md` — Terminal UI guide

## Repository Structure (Authoritative Map)

### Core Library: `code/helix/`

**Partition extraction:**
- `partitions.py` — Extract ReLU regions, build incidence B_k, compute masses τ_k
- `sparse.py` — Memory-efficient parent pointers ↔ dense incidence conversion

**Information flow:**
- `cp.py` — Build CP map operator V, verify unitality/coisometry/PSD
- `diagnostics.py` — High-level metrics (region counts, mass consistency, anisotropy)

**Mixing dynamics:**
- `ulam.py` — Standard Ulam-Perron-Frobenius with barycentric sampling
- `ulam_tensor.py` — Tensor train decomposition for high-dimensional systems (d ≥ 3)

**Topology & invariants:**
- `topology.py` — Persistent homology (Betti numbers, lifetime analysis)
- `ktheory.py` — K-theory invariants (Smith normal form, Hodge decomposition)

**Architecture adapters:**
- `architectures/vit_adapter.py` — Vision Transformer attention as gauge fields
- `trajectory.py` — O(log n) trajectory compression via importance sampling

**User interfaces:**
- `cli.py` — Command-line interface with interactive menu
- `tui.py` — Terminal UI with live visualization (requires textual)
- `plotting.py` — Matplotlib-based diagnostic plots

**Integration:**
- `env_api.py` — Stable API for verification environments (external contract)

### Examples: `code/examples/`

- `helix_demo.py` — Basic two moons classification demo
- `tensor_train_demo.py` — High-dimensional Ulam analysis

### Tests: `code/tests/`

Comprehensive coverage including:
- `test_partitions.py`, `test_cp.py`, `test_ulam.py` — Core functionality
- `test_ulam_tensor.py` — Tensor train decomposition
- `test_vit_adapter.py` — Vision Transformer analysis
- `test_llm_judge.py` — LLM evaluation integration

### Environments: `environments/`

- `helixenv/` — Verifiers-compatible AF partition environment
  - `af_partition/env.py` — Main environment implementation
  - `llm_judge.py` — Physics-aware LLM evaluation
- `bixbench/` — Additional benchmark environment

### Documentation: `docs/`

See "Documentation Structure" section above for navigation guide.

### Utilities

- `ascii-animations/` — ASCII art frames for CLI banner
- `./helix` — Local CLI runner (no installation needed)

## Key Mathematical Components (Implementation Guide)

### 1. AF Partitions (`partitions.py`)

**What it does:** Extract polyhedral regions from ReLU activations

**Key functions:**
- `extract_partitions(model, X)` → Returns incidence matrices B_k and masses τ_k
- `region_counts(B_list)` → Count regions per layer
- `mass_consistency_errors(B_list, tau_list)` → Verify probability conservation

**Memory optimization:** Use parent pointers (sparse.py) for O(n) instead of O(n²)

### 2. CP Maps (`cp.py`)

**What it does:** Build completely positive map operators and verify health

**Key functions:**
- `build_V_from_incidence(B, tau_prev, tau_cur)` → Construct operator V
- `sanity_check_ucp(V)` → Returns unitality error, coisometry error, PSD violation

**Numerical stability:** Handles zero masses, adds epsilon for stability

### 3. Ulam Operators (`ulam.py`, `ulam_tensor.py`)

**Standard Ulam (`ulam.py`):**
- `ulam_pf(flow_fn, bounds, bins_per_dim, samples_per_cell)` → Transfer matrix P
- `spectral_gap(P)` → Returns 1 - |λ₂|

**Tensor Train Ulam (`ulam_tensor.py`):**
- For high-dimensional systems (d ≥ 3)
- Complexity: O(d × bins² × r²) instead of O(bins^d)
- Auto-detects optimal rank r

### 4. Topology (`topology.py`)

**What it does:** Compute persistent homology (Betti numbers, lifetimes)

**Key functions:**
- `compute_persistent_homology(X, maxdim=2)` → Returns β₀, β₁, β₂ and lifetimes
- Automatic fallback if ripser not installed

### 5. K-Theory (`ktheory.py`)

**What it does:** Compute algebraic invariants of refinement systems

**Key functions:**
- `smith_normal_form(B)` → Canonical form revealing torsion
- `hodge_decomposition(B)` → Decompose into exact/coexact/harmonic parts

### 6. Environment API (`env_api.py`)

**Purpose:** Stable external contract for verification environments

**Key function:**
- `extract_af_metrics(model, X, ...)` → Returns comprehensive AFMetrics object

**Important:** Maintain backward compatibility—this is used by external systems

## Coding Guidelines (Critical Principles)

### 1. Surgical Changes Only

**Why:** Helix is production-ready with external dependencies (Verifiers, research code)

**Practice:**
- Don't rename/move modules without explicit instruction
- Preserve public API signatures
- `env_api.py` is an external contract—maintain backward compatibility

### 2. Numerical Stability First

**Why:** Real networks have edge cases (zero masses, singular matrices, numerical errors)

**Practice:**
- Always handle zero masses (add epsilon, check before division)
- Use vectorized NumPy operations (faster and more stable)
- Test with degenerate cases (untrained networks, single-region layers)

**Example pattern:**
```python
# Bad: can divide by zero
V[p, j] = np.sqrt(tau_k[j] / tau_prev[p])

# Good: handle zero mass
eps = 1e-12
V[p, j] = np.sqrt(tau_k[j] / (tau_prev[p] + eps))
```

### 3. Optional Dependencies with Fallbacks

**Why:** Core library should work with minimal dependencies

**Practice:**
- `torch`, `sympy`, `matplotlib`, `textual`, `tqdm` are optional
- Always provide graceful fallbacks
- Document which features require which extras

**Example pattern:**
```python
try:
    import torch
except ImportError:
    torch = None

def extract_partitions(model, X):
    if torch is None:
        raise RuntimeError("PyTorch required. Install with: pip install torch")
    # ... implementation
```

### 4. Comprehensive Testing

**Why:** Mathematical code is subtle; bugs hide in edge cases

**Practice:**
- Every new component needs tests in `code/tests/`
- Test: normal cases, edge cases, error handling
- Run `pytest -q` before committing

### 5. Progress Feedback for Long Operations

**Why:** Users need to know the code is working, not frozen

**Practice:**
- Add `show_progress=False` parameter to expensive functions
- Use tqdm when available, fallback to periodic prints
- Example: persistent homology, tensor train decomposition

### 6. Security: Never Hardcode API Keys

**Why:** Keys in code = security breach

**Practice:**
- Use environment variables: `os.getenv("OPENAI_API_KEY")`
- Document setup in `docs/api-key-setup.md`
- Fail gracefully if key missing

### 7. Memory Efficiency

**Why:** Large networks generate huge incidence matrices

**Practice:**
- Prefer parent pointers (O(n)) over dense B_k (O(n²))
- Use tensor trains for high-dimensional Ulam (d ≥ 3)
- Implement O(log n) trajectory compression when needed

### 8. Documentation Discipline

**Why:** Code teaches the math; docs make it accessible

**Practice:**
- Update `docs/` when adding features
- Write pedagogical docstrings (why, not just what)
- Keep examples runnable (test them!)

### 9. Code Quality

**Before committing:**
```bash
ruff check .    # Linting
pytest -q       # Tests
```

## Common Development Tasks (Playbook)

### Adding a New Diagnostic

**Files to modify:**
1. `code/helix/diagnostics.py` — Implement the metric
2. `code/helix/plotting.py` — Add visualization
3. `code/helix/cli.py` — Wire into CLI if user-facing
4. `code/tests/test_diagnostics.py` — Add tests

**Example:** Adding "region entropy" metric

### Extending Partition Extraction

**Files to modify:**
1. `code/helix/partitions.py` — Core extraction logic
2. `code/helix/sparse.py` — Ensure sparse helpers still work
3. `code/tests/test_partitions.py` — Test new functionality

**Critical:** Verify parent pointer ↔ dense B_k round-trip still works

### High-Dimensional Ulam Analysis

**For d ≥ 3 dimensions:**
1. Use `code/helix/ulam_tensor.py` (tensor train decomposition)
2. Benchmark with `code/scripts/benchmark_tensor_train.py`
3. Complexity: O(d × bins² × r²) vs O(bins^d)

### Vision Transformer Analysis

**Files:**
- `code/helix/architectures/vit_adapter.py` — Attention extraction
- Test attention weights as gauge field connections
- Extract topological features from attention patterns

### LLM Judge Integration

**Files:**
- `environments/helixenv/llm_judge.py` — Physics-aware evaluation
- Add calibration examples for rubric-based scoring
- Document API usage in `docs/api-key-setup.md`

### Creating New Environments

**Steps:**
1. Work in `environments/your_env/`
2. Use `env_api.py` helpers for stable metric extraction
3. Integrate with Verifiers if applicable
4. Document in `docs/environments.md`

### Adding ASCII Animations

**Steps:**
1. Create frame files in `ascii-animations/`
2. Integrate via `code/helix/cli.py` for banner/menu
3. Keep frames small (terminal-friendly)

## Quick Verification Commands

### Basic Functionality

```bash
# Interactive menu (easiest start)
./helix

# Direct CLI analysis
./helix --no-train --ulam-bins 25 --ulam-samples-per-cell 4 --plot --no-show --save-prefix helix_out

# Run demo
python code/examples/helix_demo.py
```

### Advanced Features

```bash
# K-theory analysis
./helix ktheory --env helixenv --samples 1024 --method hodge --show-progress

# Environment with LLM judge
helix helixenv --enable-llm-judge --samples 1024 --width 24

# Tensor train demo (high-dimensional)
python code/examples/tensor_train_demo.py

# Performance benchmarking
python code/scripts/benchmark_tensor_train.py --quick
```

### Testing & Quality

```bash
# Run all tests
pytest code/tests/ -v

# Quick test suite
pytest -q

# Linting
ruff check .

# Specific test modules
pytest code/tests/test_ulam_tensor.py -v
pytest code/tests/test_vit_adapter.py -v
pytest code/tests/test_llm_judge.py -v
```

### Terminal UI

```bash
# Install TUI extras
pip install -e .[tui]

# Launch interactive TUI
helix tui
```

## Environment Integration (Verifiers/Prime)

### helixenv

**Purpose:** Verifiers-compatible AF partition diagnostics environment

**Identifier:** `helix/af_partition:v0`

**Key features:**
- MCQ prompts with deterministic scoring
- Optional LLM judge (physics-aware evaluation)
- Multi-level AF metrics (partitions, CP maps, topology, K-theory)

**Usage:**

```python
# Programmatic
from environments.helixenv import load_environment
env = load_environment()

# CLI
helix helixenv --samples 1024 --noise 0.05 --width 24 --epochs 80
```

**Arguments:**
- `samples`, `noise`, `width`, `epochs` — Dataset and architecture
- `mass_weight`, `wasted_weight` — Reward function weights
- `enable_llm_judge`, `llm_judge_model` — LLM evaluation
- `show_progress` — Progress tracking for long operations

**Output:** Comprehensive AFMetrics with:
- Partition structure (regions, masses, consistency)
- CP map health (unitality, coisometry, PSD)
- Topology (persistent homology, Betti numbers)
- Capacity metrics (singular values, plasticity)
- Spectral analysis (mixing dynamics)
- K-theory invariants (torsion, Hodge decomposition)

**API Configuration:**
- Set `OPENAI_API_KEY` environment variable
- See `docs/api-key-setup.md` for detailed setup

## Important Notes for AI Assistants

### What Helix Is

✅ **Production-ready diagnostic toolkit** with:
- Comprehensive test coverage (100+ tests)
- External integrations (Verifiers, research projects)
- Stable API contracts (`env_api.py`)
- Pedagogical documentation

### Development Philosophy

✅ **Incremental improvements** — Build on existing foundation
✅ **Pedagogical focus** — Code and docs teach the math
✅ **Numerical stability** — Handle edge cases gracefully
✅ **Backward compatibility** — Don't break external dependencies

### What NOT to Do

❌ Don't restructure the codebase without explicit instruction
❌ Don't delete top-level files (e.g., `uses.md`, `CONTRIBUTING.md`)
❌ Don't break `env_api.py` contract (external systems depend on it)
❌ Don't hardcode API keys or sensitive data

### When in Doubt

1. Read the existing code—it's pedagogical by design
2. Check `docs/concepts.md` for mathematical foundations
3. Look at tests for usage examples
4. Ask for clarification before major changes

## 2025 Enhanced Capabilities

**Tensor Train Ulam:**
- High-dimensional systems (d ≥ 3)
- O(d × bins² × r²) complexity vs O(bins^d)
- Auto-rank detection and optimization

**ViT Attention Analysis:**
- Attention weights as gauge field connections
- Topological feature extraction
- Partition structure from attention patterns

**LLM Physics Judge:**
- OpenAI-integrated evaluation
- Physics-aware rubrics
- Secure API key management
- Calibration examples and fallback heuristics

**Comprehensive Testing:**
- 100+ unit tests
- Full error handling coverage
- Integration tests for all components

**Memory Optimization:**
- O(log n) trajectory compression
- Sparse matrix operations
- Efficient incidence representations

**Progress Integration:**
- All expensive operations support progress tracking
- Graceful tqdm fallbacks
- User-friendly feedback
