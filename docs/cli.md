# CLI Guide

## Quick start

```bash
./helix                 # interactive menu
./helix demo            # direct demo run
./helix --non-interactive --no-train \
  --ulam-bins 25 --ulam-samples-per-cell 4 \
  --plot --no-show --save-prefix helix_out
```

Notes:
- The repo-root `./helix` launcher works without installation.
- On Windows, use `helix.cmd` from PowerShell or Command Prompt.
- PyTorch is required for demo, analyze, and helixenv flows.

## Demo flags (no subcommand)

Common options:
- `--samples`, `--noise`, `--seed`
- `--width` (overrides `--widths`), `--widths`, `--epochs`, `--no-train`
- `--ulam-bins`, `--ulam-samples-per-cell`, `--no-ulam`
- `--plot`, `--no-show`, `--save-prefix`
- `--animate-forward` (TTY animation before summary)

Outputs include region counts, mass consistency, CP diagnostics, and a Ulam spectral gap (plots show eigenvalue magnitudes).

## Analyze a custom model and/or dataset

```bash
./helix analyze \
  --model-module path/to/your_model.py \
  --model-func build_model \
  --model-kwargs '{"in_dim": 2, "widths": [32,32], "out_dim": 2}' \
  --weights path/to/state_dict.pth \
  --data-x path/to/X.npy \
  --data-y path/to/y.npy \
  --no-ulam --no-train

# Minimal: uses a built-in MLP and two-moons data
./helix analyze --no-ulam --no-train
```

Key flags:
- `--model-module`, `--model-func`, `--model-kwargs`, `--weights`
- `--data-x`, `--data-y`
- `--d-out`, `--width`, `--epochs`, `--no-train`
- `--ulam-bins`, `--ulam-samples-per-cell`, `--no-ulam`, `--seed`
- `--allow-pickled-arrays`, `--allow-pickled-weights` (only when trusted)

## Helix environments

The Verifiers-style environment is available via `helix helixenv`:

```bash
./helix helixenv --samples 3000 --noise 0.05 --widths 32,16,16 --epochs 80
./helix helixenv --data-x data.npy --model-module my_model.py --weights model.pt --show-config
```

Run `./helix helixenv --help` for the full flag list.

## K-theory analysis

K-theory analysis is driven by the `environments/ktheory` environment:

```bash
./helix ktheory --data-x data.npy
./helix ktheory --data-x data.npy --model-module my_model.py \
  --weights model.pt --method hodge --tolerance 1e-8 \
  --max-depth 6 --save-report ktheory_report.json
```

Key flags:
- `--data-x` (required)
- `--model-module`, `--model-func`, `--weights`, `--width`
- `--method` (`hodge`, `smith`, `spectral`), `--tolerance`, `--max-depth`
- `--save-report` (JSON), `--plot`

## Other commands

- `helix tui` or `helix-tui` for the Textual interface.
- `helix reveals` for a lightweight ASCII showcase of Helix invariants.
