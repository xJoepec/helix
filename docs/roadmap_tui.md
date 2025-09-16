## Helix TUI Roadmap

### Goal
Provide an interactive Textual-based TUI for the existing CLI demo to tweak parameters and visualize computed metrics in-console.

### Implemented (v0)
- Textual TUI in `helix.tui` with inputs:
  - samples, noise, seed, width, epochs, no_train
  - ulam_bins, ulam_samples_per_cell
- Output panel prints:
  - AF region counts, mass consistency L1 errors
  - CP checks (unital, coisometry, PSD violation)
  - Ulam spectral gap and top |eigs|
- CLI integration: `helix tui` and standalone script `helix-tui`
- Packaging: optional extra `[tui]` (installs `textual` and `torch`)
- Docs: README section on TUI install/usage

### Next (v0.1)
- Async background compute with a progress indicator
- Persist/restore last used parameters
- Export JSON of results from TUI

### Nice-to-have
- Inline plots via braille/ASCII sparklines for small vectors
- Live training loss ticker when `no_train=0`
- Colorized alerts when constraints are violated (e.g., PSD)


