% helixenv

A lightweight verifiers environment that quizzes core Helix concepts via multiple‑choice questions (MCQ). Useful for smoke‑testing agents and pipelines wired to `verifiers`.

- Environment ID: `helixenv`
- Short description: MCQ questions about AF partitions, CP maps, Ulam PF, and Helix diagnostics
- Tags: helix, mcq, single-turn

## Quickstart

Run a small evaluation locally:

```bash
uv run vf-eval helixenv -a '{"max_episodes": 5}' -s
```

If using your own venv instead of `uv`, install the env package (pulls in `verifiers`):

```bash
pip install -e environments/helixenv
vf-eval helixenv -a '{"max_episodes": 5}' -s
```

CLI/TUI wrappers (reuse core Helix tools):

```bash
# Demo: region counts, mass consistency, CP checks, Ulam PF (plots optional)
python -m environments.helixenv.cli demo --samples 4000 --noise 0.07 --plot

# TUI: interactive exploration (requires extras)
pip install '.[tui]'
python -m environments.helixenv.cli tui

# Or, after installing this env package:
helixenv-cli demo --samples 4000 --noise 0.07 --plot
helixenv-cli tui

# Pro TUI (advanced presets, tables, exports)
python -m environments.helixenv.cli pro
helixenv-cli pro

Features
- Presets: Quick demo, High-res Ulam, K-theory only
- Tables: Regions, Mass, CP, Ulam, Sparse, K-theory
- Compare: Add runs, side-by-side tables, Export Compare (JSON/CSV)
- Sweeps→Compare: configure `sweep_seeds`, `sweep_widths`, then run
- Exports: JSON/CSVs to `export_dir`
- API Keys: set `api_key_var`, `api_key`, optional `api_base_url`

Key bindings
- r: Run current config
- w: Sweep→Compare
- a/m/c: Add/Export/Clear Compare
- e/x: Export JSON / Export CSVs
- k: Set API key env vars
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as JSON.
- The environment is self-contained and does not require external datasets.
- The CLI/TUI wrappers forward to the `helix` package in `code/helix/`, exposing the
  use cases described in `uses.md` without duplicating logic here.

## Environment Arguments

- `mode` ("zero_shot" | "agentic", default "zero_shot"): execution mode
- `answer_mode` ("mcq" | "open", default "mcq"): answer format
- `max_episodes` (int | null, default null): limit number of items
- `shuffle_options` (bool, default true): shuffle MCQ options with a seeded RNG
- `with_refusal` (bool, default false): append “I don’t know” as option E
- `seed` (int, default 42): seed for deterministic shuffling
- `use_think` (bool, default false): use ThinkParser for chain-of-thought parsing
- `max_turns` (int, default 10): agentic: assistant replies before stopping
- `system_prompt` (str | null): override the default system prompt

## Metrics

- `reward`: 1.0 for correct letter, else 0.0

## Example

```bash
env=helixenv episodes=5 mode=zero_shot answer_mode=mcq
reward/avg=0.60 std=0.49
```
