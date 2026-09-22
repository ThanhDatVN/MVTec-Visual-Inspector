"""Data layer.

`core`, `layout`, `splits` and `integrity` are torch-free by design: split
integrity is the foundation every later number rests on, so it must be
verifiable in CI without a GPU or a 2.5 GB wheel. Torch-dependent wrappers live
in `torch_datasets` and are imported lazily.
"""

from .core import ANOMALOUS, NORMAL, DatasetIndex, Sample, discover_category, discover_split
from .integrity import LeakReport, audit
from .layout import LAYOUTS, MVTEC_AD, MVTEC_AD2, SYNTHETIC, DatasetLayout, get_layout
from .splits import SplitAssignment, carve_validation, ensure_validation

__all__ = [
    "ANOMALOUS",
    "LAYOUTS",
    "MVTEC_AD",
    "MVTEC_AD2",
    "NORMAL",
    "SYNTHETIC",
    "DatasetIndex",
    "DatasetLayout",
    "LeakReport",
    "Sample",
    "SplitAssignment",
    "audit",
    "carve_validation",
    "discover_category",
    "discover_split",
    "ensure_validation",
    "get_layout",
]
