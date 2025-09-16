# Troubleshooting

Common
- Module not found: Use `./helix` or `python -m helix`, or install in a venv (`pip install -e .`).
- Missing Matplotlib: `python3 -m pip install matplotlib`.
- SymPy required for K‑theory: `python3 -m pip install sympy`.

Performance tips
- Prefer parent pointers over dense incidence where possible.
- Reduce `--ulam-bins` and `--ulam-samples-per-cell` for faster Ulam runs.

Reproducibility
- Set seeds for data generation; Ulam uses a fixed RNG unless provided.

