"""Helix computational toolkit bridging operator algebras and ML diagnostics.

Modules
-------
- partitions: extract ReLU partitions, incidence matrices, and masses
- cp: build unital CP embeddings from incidence; sanity checks
- ulam: Ulam–Perron–Frobenius discretization and spectral gap
- ktheory: Smith normal form helpers and K-invariant readouts
- diagnostics: convenience metrics for training-time monitoring
"""

from .partitions import (
    PartitionLevel,
    AFExtraction,
    extract_partitions,
)
from .cp import (
    build_V_from_incidence,
    cp_embed_apply,
    sanity_check_ucp,
)
from .ulam import (
    ulam_pf,
    spectral_gap,
)
from .ktheory import (
    smith_normal_form_Z,
    k_invariants_from_B,
)
from .diagnostics import (
    region_counts,
    mass_consistency_errors,
    cumulative_anisotropy,
)
from .sparse import (
    parents_from_B,
    B_from_parents,
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

