# Getting Started

Install
```
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .
# Optional
python3 -m pip install torch sympy matplotlib
```

Quick CLI
```
./helix --no-train --ulam-bins 25 --ulam-samples-per-cell 4 --plot --no-show --save-prefix helix_out
```

Interactive TUI
```
python3 -m pip install -e .[tui]
helix tui  # or: helix-tui
```

Programmatic API
See README.md for a minimal snippet; full API surface in docs/api.md.

Outputs
- AF: region counts n_k, mass consistency ‖τ_{k-1} − B_k τ_k‖₁
- CP: unitality ‖Φ(I) − I‖_F, coisometry proxy ‖V V* − I‖_F, PSD min‑eig violation
- Ulam: top |λ| of P^T and spectral gap 1 − |λ₂|

