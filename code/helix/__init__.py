"""Helix computational toolkit bridging operator algebras and ML diagnostics.

Modules
-------
- partitions: extract ReLU partitions, incidence matrices, and masses
- cp: build unital CP embeddings from incidence; sanity checks
- ulam: Ulam–Perron–Frobenius discretization and spectral gap
- ktheory: Smith normal form helpers and K-invariant readouts
- diagnostics: convenience metrics for training-time monitoring
"""

from .cp import (
    build_V_from_incidence,
    cp_embed_apply,
    sanity_check_ucp,
)
from .diagnostics import (
    cumulative_anisotropy,
    mass_consistency_errors,
    region_counts,
)
from .ktheory import (
    k_invariants_from_B,
    smith_normal_form_Z,
)
from .partitions import (
    AFExtraction,
    PartitionLevel,
    extract_partitions,
)
from .sparse import (
    B_from_parents,
    parents_from_B,
)
from .ulam import (
    spectral_gap,
    ulam_pf,
)

__all__ = [
    # partitions
    "PartitionLevel",
    "AFExtraction",
    "extract_partitions",
    # cp
    "build_V_from_incidence",
    "cp_embed_apply",
    "sanity_check_ucp",
    # ulam
    "ulam_pf",
    "spectral_gap",
    # ktheory
    "smith_normal_form_Z",
    "k_invariants_from_B",
    # diagnostics
    "region_counts",
    "mass_consistency_errors",
    "cumulative_anisotropy",
    # sparse helpers
    "parents_from_B",
    "B_from_parents",
]

