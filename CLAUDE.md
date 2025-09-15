# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Project Helix formalizes the "deep networks as manifold sculptors" intuition using operator algebras. It provides a mathematical framework bridging machine learning geometry with rigorous operator-algebraic tools (AF algebras, crossed products, CP maps, groupoids, Morita equivalence).

## Core Architecture

The project establishes a dictionary between ML concepts and operator algebra:
- **Data manifold X** → Observable algebra `A₀ = C(X)`
- **Network layers** → *-endomorphisms and CP maps
- **ReLU partitions** → AF inductive limits via Bratteli diagrams
- **Invertible blocks** → Crossed products `C(X) ⋊ Z/R`
- **Width/channels** → Matrix stabilization and Morita equivalence
- **Training** → Deformation within Morita class toward simpler representatives

## Key Mathematical Components

1. **AF Algebras**: Track ReLU partition refinement through inductive limits
   - Incidence matrices `Bₖ` encode cell parent-child relationships
   - Dimension group `K₀(A_AF)` captures asymptotic refinement statistics

2. **CP Maps & Stinespring**: Model nonlinear activations
   - Dilation: `Φ(X) = V* π(X) V` recovers linearity on larger space
   - Unital CP maps preserve traces and positivity

3. **Crossed Products**: Encode dynamics of invertible/flow layers
   - `C(X) ⋊_α Z` for discrete iterations
   - Spectral gaps and KMS states track mixing properties

4. **Groupoids**: Handle noninvertible endomorphisms
   - Deaconu-Renault construction for local homeomorphisms
   - K-theory via Smith normal form of `I - B^T`

## Implementation Roadmap

Next steps for code development (to be placed in `code/` directory):

1. **Partition Extraction Module**
   - Extract ReLU activation patterns
   - Build incidence matrices `Bₖ` and compute masses `τₖ`
   - Verify consistency: `τₖ₋₁ = Bₖ τₖ`

2. **CP Map Construction**
   - Implement `Vₖ = D(τₖ₋₁)^(-1/2) Bₖ D(τₖ)^(1/2)`
   - Verify unitality and positive-semidefinite preservation

3. **Flow Analysis Tools**
   - Ulam discretization for Perron-Frobenius operators
   - Compute spectral gaps `1 - |λ₂(P)|`
   - Track Jacobian statistics (log-det, singular values)

4. **K-Theory Computation**
   - Smith normal form wrapper for `I - B^T`
   - Extract `K₀ ≅ coker(I - B^T)` and `K₁ ≅ ker(I - B^T)`

## Key Files

- **overview.md**: High-level project description, workflow, and diagnostics
- **Notes-1.md**: Conceptual bridge between ML manifolds and operator algebras
- **Notes-2-Morita.md**: Detailed formalism with compute-ready recipes and example code snippets
- **helix-latex**: Full paper draft with proofs and references (revtex4-2)
- **code/helix**: Core Python library (partitions, CP maps, Ulam PF, K-theory, diagnostics, sparse helpers)
- **code/examples/helix_demo.py**: End-to-end demo script
- **helix**: CLI entrypoint (adds `code/` to `sys.path` and runs `helix.cli`)

## Development Notes

- The project now includes a working implementation (partition extraction, CP diagnostics, Ulam with barycentric sampling, plotting, CLI, and minimal tests).
- Focus on building computational tools that bridge abstract operator algebra with practical ML diagnostics.
- Maintain rigor while ensuring computability for real neural networks.

## CLI & Plots

- Run: `./helix --plot --ulam-bins 25 --ulam-samples-per-cell 4 --save-prefix helix_out`
- Outputs: `helix_out_regions.png`, `helix_out_mass_consistency.png`, `helix_out_cp.png`, `helix_out_ulam.png`