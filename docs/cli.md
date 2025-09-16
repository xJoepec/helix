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

