# LLM judge API-key setup

The Helix LLM judge evaluates structured `AFMetrics` through
`LLMJudge.evaluate_af_metrics`. It does not accept generic prompt/response pairs.

## Offline heuristic evaluation

An API key is optional when heuristic fallback is enabled (the default). This is useful for
local development and deterministic tests:

```python
from environments.helixenv.llm_judge import LLMJudge
from helix.env_api import extract_af_metrics

metrics = extract_af_metrics(
    model,
    samples,
    compute_ph=True,
    compute_spectral_gap=True,
)

judge = LLMJudge(api_key=None, fallback_to_heuristic=True)
result = judge.evaluate_af_metrics(
    metrics,
    step=training_step,
    context={"run_id": run_id},
)

print(result.overall_label)
print(result.rubric_scores)
print(result.explanation)
```

The heuristic uses the deepest AF level and scores wave coherence, gauge invariance,
ergodic mixing, topological robustness, and information preservation. If no levels are
available, it returns an unstable result with the `empty_partitions` consistency flag.

## API-backed evaluation

Create an API key with your provider and expose it as `OPENAI_API_KEY` in the shell that
runs Helix. Do not put the key in source code or commit it to an environment file.

PowerShell:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
uv run python your_script.py
```

Bash-compatible shells:

```bash
export OPENAI_API_KEY="your-api-key"
uv run python your_script.py
```

Then construct the judge without embedding the secret:

```python
from environments.helixenv.llm_judge import LLMJudge

judge = LLMJudge(
    model_name="your-chat-model",
    temperature=0.1,
    max_tokens=500,
    enable_calibration=True,
    fallback_to_heuristic=True,
)

result = judge.evaluate_af_metrics(
    metrics,
    step=training_step,
    context={"experiment": "af-stability"},
)
```

`LLMJudge` reads `OPENAI_API_KEY` when `api_key` is omitted. With fallback enabled, an
unavailable key, API error, observation error, or response-parsing error falls back to the
same structured heuristic evaluation. Set `fallback_to_heuristic=False` when failures must
surface to the caller instead.

## Rubric and calibration

`PhysicsRubric.default_rubric()` defines the category thresholds and weights. Supply a
custom `PhysicsRubric` through the `rubric` argument when an experiment requires different
thresholds:

```python
from environments.helixenv.llm_judge import LLMJudge, PhysicsRubric

rubric = PhysicsRubric.default_rubric()
judge = LLMJudge(rubric=rubric, enable_calibration=True)
```

With calibration enabled, the prompt includes stable and unstable AF examples before the
current structured observation. The model response must be a JSON object with:

- `overall_label`: `A`, `B`, or `C`
- `confidence`: a number from `0.0` to `1.0`
- `rubric_scores`: ratings for the five rubric categories
- `explanation`: concise reasoning
- `consistency_flags`: optional list of detected issues

The parser accepts a plain JSON object or an object wrapped in explanatory text or a fenced
code block.

## Troubleshooting

- **No key and fallback disabled:** set `OPENAI_API_KEY`, pass `api_key` explicitly from a
  secret manager, or enable heuristic fallback.
- **Authentication or API failure:** verify the key in the current shell. With fallback
  enabled, inspect `result.model_used`; `heuristic_fallback` means no model result was used.
- **Unexpected heuristic result:** inspect the deepest `AFLevelMetrics`, especially CP
  diagnostics, spectral gap, persistent homology, and mass error. Those values drive the
  rubric scores.
- **Malformed model output:** keep the requested nested JSON response shape. Parsing errors
  are raised when fallback is disabled and otherwise trigger heuristic evaluation.
