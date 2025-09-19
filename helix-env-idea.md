Operator-Algebra Helix Environment (Verifier/Prime Roadmap)
===========================================================

This note captures a soup-to-nuts plan for turning the Helix operator-algebra diagnostics into Verifiers-compatible training/eval environments that look and feel like the Prime environments (e.g., `bixbench`, `hle`). The goal is to let a lab run the full AF/CP/flow workflow on a trained PyTorch model, and then expose the resulting health signals through Verifiers so Prime-rl (or any Verifiers-compatible trainer) can optimise agents against them.

0. Setup (researcher mindset)
-----------------------------
- **Inputs:** Trained `nn.Module`, representative dataset/sample, metadata about architectural symmetry or residual flows if available.
- **Libraries:** PyTorch, NumPy/SciPy, SymPy (Smith Normal Form), plotting stack (matplotlib/plotly), Verifiers + Prime runtime.
- **Outcome:** A repeatable pipeline that produces operator-algebra invariants and feeds them into an RL/eval loop.

1. Partition Extraction (AF algebra stage)
-----------------------------------------
- Run the dataset through the model, capturing ReLU activation sign patterns per layer.
- Each unique pattern defines a region \( P^{(k)}_j \); collect them into depth-indexed partitions.
- Build incidence matrices \( B_k \in \{0,1\}^{n_{k-1}\times n_k} \) where each child sits under exactly one parent.
- Compute region counts \( n_k \); track \( \frac{1}{k}\log n_k \) as a combinatorial entropy proxy.
- **Alert:** Rapidly growing \( n_k \) with highly concentrated mass indicates wasted expressivity/overfit risk.

2. Mass Consistency & Traces
----------------------------
- Estimate masses \( \tau_k(j) = \mu(P^{(k)}_j) \) from empirical frequencies.
- Verify the trace condition \( \tau_{k-1} = B_k \tau_k \).
- **Metric:** \( \lVert \tau_{k-1} - B_k \tau_k \rVert_1 \) as a gradient/optimiser health signal.
- Interpret the \( (\tau_k) \) tower as a trace on the AF algebra—the model’s probability ledger.

3. CP Maps & Diagnostics
------------------------
- Construct unital CP embeddings \( V_k = D(\tau_{k-1})^{-1/2} B_k D(\tau_k)^{1/2} \) and \( \Phi_k(X) = V_k^* X V_k \).
- Run checks: \( \Phi_k(I) = I \) (unitality) and PSD preservation on random PSD probes.
- **Interpretation:** CP violations imply ill-conditioning, dead regions, or numerical drift.

4. Invariants (K-theory & Morita classes)
-----------------------------------------
- Assemble the Bratteli diagram from \( (B_k) \).
- For stationary/periodic patterns compute Smith Normal Form of \( I - B^\top \) to obtain \( K_0 \cong \text{coker}(I - B^\top) \) and \( K_1 \cong \ker(I - B^\top) \).
- **Morita lens:** Two nets share computational essence when their AF algebras match in K-theory + trace; width tweaks that preserve these invariants leave function intact.

5. Flow / Residual Blocks (crossed products)
-------------------------------------------
- Treat near-invertible residual blocks as diffeomorphisms \( \varphi \).
- Build Ulam–Perron–Frobenius operator \( P \) by discretising and pushing mass.
- Track spectral gap \( 1 - |\lambda_2| \); widening gaps suggest better mixing and separation.
- Interpret \( C(X) \rtimes_{\alpha} \mathbb{Z} \) as the algebraic footprint of time evolution.

6. Noninvertible Maps (groupoids)
---------------------------------
- For many-to-one maps form the equivalence relation groupoid over colliding points.
- The associated Cuntz–Pimsner algebra captures duplication/merging behaviour.
- Numerically infer structure directly from incidence matrices.

7. Equivariance / Symmetry Audits
---------------------------------
- For known symmetry group \( G \), check \( F(gx) \approx gF(x) \) empirically.
- Operator algebra model: representation of \( C(X) \rtimes G \).
- Track violation decay through training.

8. Monitoring & Production Use
------------------------------
- **Primary gauges:** region counts, mass consistency error, CP unitality/positivity, PF spectral gap.
- **Alarms:** gap collapse (distribution shift), exploding \( n_k \) with collapsing traces (overfit), CP violations (instability).

9. Educational / Research Use
-----------------------------
- Visualise Bratteli diagrams, mass flows, eigenvalue spectra.
- Demonstrate how regions split/merge and carry data mass.
- Compare theoretical predictions to live measurements from trained nets.

10. Scientific Computing & Quantum Analogy
------------------------------------------
- Treat neural ODEs via PF operators to study stability/chaos.
- CP maps + Stinespring dilations mirror quantum channels; Helix bridges ML with quantum-inspired design.

Deliverable: Diagnostic Report
------------------------------
- Region/partition stats with combinatorial entropy.
- Mass flow tables and trace residuals.
- CP map health (unitality, PSD slack, \( \lVert V V^* - I \rVert_F \)).
- Bratteli/K-theory invariants.
- PF spectral analysis and interpretive commentary (e.g., “stable but over-partitioned”).

Turning the Workflow into Verifiers Environments
===============================================

Why Verifiers + Prime?
----------------------
- Verifiers supplies Env abstractions, async rollouts, GRPO-style training, and doubles as an eval harness.
- Prime (prime-rl, prime-cli) offers environment hubs, distributed training (FSDP, decentralised), and task management. `prime-environments` is the template we mirror.

Proposed repo layout (`operator-algebra-environments`)
-----------------------------------------------------
```
operator-algebra-environments/
  oa_envs/
    __init__.py
    registry.py                # registers env IDs with Verifiers/Prime hub
    af_partition/
      env.py                    # AF traces & Bratteli tasks
      specs.py                  # observation/action/reward schemas
      tests/
    cp_dilation/
      env.py                    # build V_k, UCP/PSD checks
    ulam_flow/
      env.py                    # PF operator & spectral rewards
    equivariance/
      env.py                    # crossed-product audits
  examples/
    train_grpo_af.py            # GRPO training using Verifiers
    eval_cp.py                  # evaluation harness
  pyproject.toml
  README.md
```

Environment contracts (minimal)
-------------------------------
- Each env subclasses `verifiers.core.Env`, implements `reset/step/render`, and registers via `@register_env` (from `verifiers`).
- Observations: numeric feature vectors (e.g., `n_k`, mass residuals, PF gaps) + context text/JSON; optional artefacts for large matrices.
- Actions: agent text/tool calls or structured JSON knobs (e.g., widen layer, adjust LR, recompute diagnostics).
- Rewards:
  - AF: \( r = -\lVert \tau_{k-1}-B_k \tau_k \rVert_1 - \lambda_1 \cdot \text{wasted regions} + \lambda_2 \cdot \Delta \text{val acc} \)
  - CP: penalise unital/PSD violations, condition numbers.
  - Ulam: reward PF gap improvements, penalise out-of-bounds mapping.
  - Equivariance: penalise max/mean symmetry violation.

Example skeleton (`oa_envs/af_partition/env.py`)
-----------------------------------------------
```
from verifiers.core import Env, Step, register_env
import numpy as np

from oa_envs.af_partition.ops import compute_incidence_and_mass

@register_env("oa/af_partition:v0")
class AFPartitionEnv(Env):
    def __init__(self, model, dataset, k_max=5, lam_overfit=0.1):
        self.model = model
        self.X = dataset
        self.k_max = k_max
        self.k = 0
        self.lam = lam_overfit
        self.state = None

    def reset(self, seed=None):
        self.k = 1
        self.state = self._compute_level(self.k)
        obs = self._obs(self.state)
        return Step(obs=obs, reward=0.0, done=False, info={})

    def step(self, action):
        self._apply_action(action)
        self.k = min(self.k + 1, self.k_max)
        self.state = self._compute_level(self.k)
        obs = self._obs(self.state)
        reward = self._reward(self.state)
        done = self.k == self.k_max
        return Step(obs=obs, reward=reward, done=done, info={"k": self.k})

    def _compute_level(self, k):
        B_k, tau_prev, tau_k, wasted = compute_incidence_and_mass(self.model, self.X, k)
        mass_err = float(np.abs(tau_prev - B_k @ tau_k).sum())
        return {
            "B_k": B_k,
            "tau_prev": tau_prev,
            "tau_k": tau_k,
            "mass_err": mass_err,
            "wasted_regions": wasted,
        }

    def _obs(self, state):
        return {
            "features": np.array([
                state["B_k"].shape[1],
                state["mass_err"],
                state["wasted_regions"],
            ], dtype=np.float32),
            "context": f"depth {self.k}/{self.k_max}",
        }

    def _reward(self, state):
        return -state["mass_err"] - self.lam * state["wasted_regions"]

    def _apply_action(self, action):
        pass
```
- Reuse existing Helix helpers (`extract_partitions`, `build_V_from_incidence`, `mass_consistency_errors`, `spectral_gap`, etc.) to implement the shared ops modules.

Pipeline integration steps
--------------------------
1. Factor Helix utilities into importable modules (`helix` already has them; expose stable APIs).
2. Implement Verifiers envs (one per task family) using those utilities.
3. Provide spec files describing observation/action spaces for docs and testing.
4. Register env IDs in `registry.py` for discovery (`oa/af_partition:v0`, etc.).
5. Add minimal tests verifying invariants (mass consistency zeroed on exact data, CP positivity holds on curated samples).
6. Supply CLI scripts under `examples/` for training (`train_grpo_af.py`) and eval (`eval_cp.py`).
7. Document usage in `README.md` + per-env cards.

Training & evaluation commands
------------------------------
```
# local dev install
pip install -e verifiers -e prime-rl -e operator-algebra-environments

# sample GRPO run
python examples/train_grpo_af.py \
  --env_id oa/af_partition:v0 \
  --policy gpt-4o-mini \
  --rollouts 64 --steps_per_update 2 \
  --reward_scale 1.0 --max_steps 5000

# scale with Prime CLI
gpip install prime-cli
prime env install https://github.com/<you>/operator-algebra-environments
prime rl train --env oa/af_partition:v0 --nodes 8 --gpus-per-node 4 \
  --trainer grpo --checkpoint s3://bucket/checkpoints/run-001

# evaluation harness
python examples/eval_cp.py \
  --env_id oa/cp_dilation:v0 --model ckpt:path/to/model --report out/report.json
```

Signals to expose per environment
---------------------------------
- **AF/ReLU:** `n_k`, \( \lVert \tau_{k-1}-B_k \tau_k \rVert_1 \), entropy of \( \tau_k \), cumulative anisotropy from \( B_{1:k} \).
- **CP/Dilation:** unitality error \( \lVert \Phi(I)-I \rVert_F \), PSD slack, Choi matrix minima, condition numbers.
- **Ulam/Flow:** PF leading eigenvalues, spectral gap, Jacobian statistics.
- **Equivariance:** max/mean violation across sampled \( g \in G \); optionally attach accuracy deltas.

Publishing checklist
--------------------
- Populate `registry.py` to register env IDs with Verifiers.
- Add env cards (`cards/oa_af_partition.md`, etc.) describing observations/actions/rewards/examples.
- Ship minimal unit tests (mass conservation sugar tests, CP positivity checks).
- Provide Prime metadata so `prime-cli` can auto-discover environments.
- Document dataset/model adapters (Torch dataloader → cached tensors) for reproducibility.
- Optional: expose Prime tool sandboxes for controlled code execution, and ship dashboards for Bratteli/CP/PF visualisations.

Next actions
------------
1. Stabilise Helix helper APIs (`helixenv.py`, `helix/cli.py` utilities) so env modules can import them cleanly.
2. Scaffold the `operator-algebra-environments` repo using the layout above.
3. Implement `af_partition` env end-to-end, including tests and example script; iterate before cloning pattern to CP/Ulam/equivariance.
4. Register environments with Prime hub, run a smoke GRPO training loop, and capture first diagnostic reports.
