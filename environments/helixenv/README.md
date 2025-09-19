# helixenv

Operator-algebra diagnostics packaged as Verifiers/Prime-compatible environments. The default AF
partition task mirrors the workflow from `helix-env-idea.md`: agents inspect layer-wise region
statistics, CP diagnostics, and Ulam transfer metrics, then classify the run as **stable**,
**capacity-wasted**, or **collapsed**.

### Overview
- **Environment ID**: `helix/af_partition:v0`
- **Short description**: Multi-level AF partition diagnostics rendered as MCQ prompts
- **Tags**: operator-algebra, diagnostics, mcq

### Quickstart
Run the CLI demo that synthesises a two-moons model, extracts diagnostics, and prints layer stats:

```bash
helix helixenv --samples 1024 --noise 0.05 --width 24 --epochs 80
```

Provide an API key for LLM-based rubric extensions (stored in `OPENAI_API_KEY` unless overridden):

```bash
helix helixenv --api-key sk-your-key --api-key-var OPENAI_API_KEY
```

Interactive mode (`helix` with no arguments) also prompts for the key using a non-echoing input. The
key is only exported for the current process, matching how `bixbench` and `hle` expect judge keys.

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

### Environment arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `samples` | int | 2000 | Synthetic dataset size when no features are provided |
| `noise` | float | 0.08 | Noise level for the two-moons generator |
| `seed` | int | 1 | RNG seed for data and training |
| `width` | int | 16 | Hidden width for the demo MLP |
| `epochs` | int | 60 | Demo training epochs |
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
  the same key handling, keeping parity with `bixbench` and `hle`.
- See `helix-env-idea.md` for the broader roadmap covering CP/Ulam/equivariance environments.
