# helixenv

Operator-algebra diagnostics packaged as Verifiers/Prime-compatible environments. Multiple environment
variants focus on different aspects of neural network operator-algebraic analysis:

- **AF Partition** (`helix/af_partition:v0`): Layer-wise region statistics and mass flow analysis
- **CP Dilation** (`helix/cp_dilation:v0`): Completely positive map health and Stinespring diagnostics
- **Ulam Flow** (`helix/ulam_flow:v0`): Perron-Frobenius operator spectral analysis

### Overview
- **Primary Environment ID**: `helix/af_partition:v0`
- **Short description**: Multi-level operator-algebraic diagnostics with diverse datasets
- **Tags**: operator-algebra, diagnostics, mcq, manifold-learning, topology

### Quickstart
Run with Verifiers (vf-eval) similar to bixbench:

```bash
uv run vf-eval helix/af_partition:v0 -a '{"max_episodes": 8}'
```

Enable the LLM judge by exporting a key (optional):

```bash
export OPENAI_API_KEY=sk-your-key
uv run vf-eval helix/af_partition:v0 -a '{"enable_llm_judge": true, "max_episodes": 8}'
```

Or run the bundled CLI demo with diverse datasets:

```bash
# Traditional two moons
helix helixenv --samples 1024 --noise 0.05 --width 24 --epochs 80

# Swiss Roll manifold (3D)
helix helixenv --dataset-type swiss_roll --samples 1024 --noise 0.02 --width 32 --epochs 100

# XOR non-linear separability test
helix helixenv --dataset-type xor --samples 800 --noise 0.08 --width 16 --epochs 60

# Concentric circles (radial structure)
helix helixenv --dataset-type circles --samples 1000 --noise 0.05 --width 20 --epochs 80
```

Interactive mode (`helix` with no arguments) also prompts for the key using a non-echoing input. The
key is only exported for the current process, matching how `bixbench` expects judge keys.

Enable the optional LLM judge when calling the Verifiers loader:

```python
from environments.helixenv import load_environment

env = load_environment(
    enable_llm_judge=True,
    llm_judge_model="gpt-4.1-mini",
    llm_judge_api_key_var="OPENAI_API_KEY",
)
```

The judge checks the assistant’s letter and emits `verdict: correct/incorrect`. Any key set through
the CLI is already available via `OPENAI_API_KEY`.

### Verifiers integration
Programmatic access mirrors other environments:

```python
from environments.helixenv import load_environment

env = load_environment(max_episodes=4)
step = env.reset()
```

Register with Verifiers/Prime hubs via:

```python
from environments.helixenv import register_helix_envs
register_helix_envs(registry.register_env)
```

### Dataset Types

The AF partition environment now supports multiple dataset types for comprehensive testing:

| Dataset Type | Dimension | Description | Best for testing |
| ------------ | --------- | ----------- | ---------------- |
| `moons` | 2D | Traditional two moons | Basic classification, moderate complexity |
| `swiss_roll` | 3D | Swiss roll manifold | Manifold learning, high-dimensional embedding |
| `circles` | 2D | Concentric circles | Radial separation, circular decision boundaries |
| `xor` | 2D | XOR pattern clusters | Non-linear separability, architectural depth |
| `s_curve` | 3D | S-shaped manifold | Smooth manifold topology, curvature effects |

Additional dataset parameters:
- `swiss_roll` supports `hole=True` for topological complexity
- `circles` supports `factor=0.8` for inner/outer circle ratio

### Environment Variants

#### AF Partition (`helix/af_partition:v0`)
Primary environment focusing on AF algebra partition analysis:
- Mass consistency tracking across layers
- Region count and wasted capacity analysis
- Combinatorial entropy evolution
- CP map health diagnostics
- Persistent homology summaries (β₀/β₁ lifetimes with ripser fallback)
- Capacity loss detection via singular-value collapse statistics
- CLI dashboard surfaces combined topology/capacity signals

#### CP Dilation (`helix/cp_dilation:v0`)
Specialized environment for completely positive map analysis:
- Unital and coisometry error tracking
- Choi matrix eigenvalue analysis
- Stinespring dilation rank estimation
- Channel capacity approximation

#### Ulam Flow (`helix/ulam_flow:v0`)
Flow dynamics analysis via Ulam discretization:
- Perron-Frobenius operator spectral gaps
- Mixing time estimation
- Jacobian determinant statistics
- Ergodicity measures

### Environment arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `samples` | int | 2000 | Synthetic dataset size when no features are provided |
| `noise` | float | 0.08 | Noise level for dataset generation |
| `seed` | int | 1 | RNG seed for data and training |
| `width` | int | 16 | Hidden width for the demo MLP |
| `epochs` | int | 60 | Demo training epochs |
| `dataset_type` | str | `moons` | Dataset type: `moons`, `swiss_roll`, `circles`, `xor`, `s_curve` |
| `dataset_kwargs` | dict | `{}` | Additional dataset-specific parameters (e.g., `{"hole": true}`) |
| `mass_weight` | float | 1.0 | Reward weight for mass-consistency error |
| `wasted_weight` | float | 0.1 | Reward weight for wasted regions |
| `max_depth` | int | 0 | Depth cap (0 means use all levels) |
| `api_key` | str | "" | Optional API key captured from the CLI and exported for rubrics |
| `api_key_var` | str | `OPENAI_API_KEY` | Environment variable that receives the key |
| `enable_llm_judge` | bool | `False` | Use an LLM-based rubric (requires API key) |
| `llm_judge_model` | str | `gpt-4.1-mini` | Model name for the judge client |
| `llm_judge_base_url` | str | `https://api.openai.com/v1` | Base URL for the judge provider |
| `llm_judge_api_key_var` | str | `OPENAI_API_KEY` | Env var name supplying the judge key |

Runtime adapters may also provide `data_x`/`data_y`, `model_module`, `model_kwargs`, and `weights`
to analyse custom networks.

### Notes
- The Verifiers loader emits multiple-choice prompts with deterministic scoring, so the API key is
  optional today. Future rubric extensions (LLM-based diagnostics, narrative rationales) will reuse
  the same key handling, keeping parity with `bixbench`.
- Multiple dataset types provide diverse testing scenarios for operator-algebraic diagnostics,
  from basic 2D classification to complex 3D manifold learning challenges.

### Future Roadmap
Planned enhancements include:
- **Persistent Homology Trajectories**: Track barcodes across training checkpoints
- **K-Theory Environment**: Bratteli diagram analysis and K₀/K₁ invariant computation
- **Equivariance Testing**: Group symmetry violation detection and monitoring
- **Visualization Dashboard (GUI)**: Live Bratteli diagrams and PF spectra beyond the CLI cards
- **Multi-architecture Support**: Vision transformers, CNNs, and graph neural networks

See `helix-env-idea.md` for detailed implementation plans.
