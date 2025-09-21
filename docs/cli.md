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

## K-theory Analysis

Analyze K-theory invariants and algebraic topology of AF partitions:

```
# Run K-theory analysis with different environments and configurations
./helix ktheory --env helixenv --samples 1024 --noise 0.05 --width 24
./helix ktheory --env custom --model-path model.pth --data-path data.npy

# Advanced K-theory options
./helix ktheory --method hodge --tolerance 1e-8 --show-progress
./helix ktheory --output-file ktheory_results.json --verbose
```

### K-theory Flags

- `--env`: Environment type (`helixenv`, `custom`)
- `--method`: Computation method (`smith` for Smith normal form, `hodge` for Hodge decomposition)
- `--tolerance`: Numerical tolerance for Hodge method (default: 1e-10)
- `--show-progress`: Display progress bars during computation
- `--output-file`: Save results to JSON file
- `--verbose`: Show detailed diagnostic information

### K-theory Output

The K-theory analysis provides:
- **Betti numbers**: Topological invariants (rank of homology groups)
- **Torsion coefficients**: Finite order elements in K-groups
- **Smith normal form**: Canonical matrix decomposition
- **Harmonic eigenvalues**: Spectral information from Hodge Laplacian
- **K₀/K₁ invariants**: Algebraic K-theory groups

This is useful for understanding the topological complexity and algebraic structure of neural network partitions.
