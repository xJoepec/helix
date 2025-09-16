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

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as JSON.
- The environment is self-contained and does not require external datasets.

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
