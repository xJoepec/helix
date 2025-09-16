# CLI Guide

Basics
```
./helix --help
./helix --no-train --ulam-bins 25 --ulam-samples-per-cell 4 --plot --no-show --save-prefix helix_out
```

Key flags
- --samples, --noise, --width, --epochs, --no-train
- --ulam-bins: grid resolution per dimension
- --ulam-samples-per-cell: random samples per grid cell (barycentric splitting)
- --plot, --no-show, --save-prefix

Outputs
- Region counts by depth, mass L1 consistency, CP diagnostics, Ulam eigen spectrum

Notes
- Requires PyTorch when running the demo (to build/train the MLP and extract partitions).
- Use the local helper `helix` at repo root to run without installing.

Custom model/data (analyze)
```
# Analyze a user model and/or dataset
./helix analyze \
  --model-module path/to/your_model.py \
  --model-func build_model \
  --model-kwargs '{"in_dim": 2, "widths": [32,32], "out_dim": 2}' \
  --weights path/to/state_dict.pth \
  --data-x path/to/X.npy \
  --data-y path/to/y.npy \
  --no-ulam --no-train

# Minimal: use built-in MLP sized to data and synthesized moons if no --data-x
./helix analyze --no-ulam --no-train
```

Flags (analyze)
- `--model-module`: Python file that defines a builder function (returns `nn.Module`).
- `--model-func`: Builder function name in that module (default `build_model`).
- `--model-kwargs`: JSON dict passed to the builder.
- `--weights`: Optional PyTorch `state_dict` to load.
- `--data-x`: Features array (`.npy/.npz/.csv`); shape `[N, d]`.
- `--data-y`: Optional labels for training (if desired).
- `--no-train`, `--epochs`: Training control (only if labels provided).
- `--ulam-bins`, `--ulam-samples-per-cell`, `--no-ulam`: Ulam settings.
