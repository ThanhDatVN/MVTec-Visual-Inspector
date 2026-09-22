"""Data layer.

`core`, `layout`, `splits` and `integrity` are torch-free by design: split
integrity is the foundation every later number rests on, so it must be
verifiable in CI without a GPU or a 2.5 GB wheel. Torch-dependent wrappers live
in `torch_datasets` and are imported lazily.
"""

from __future__ import annotations

from pathlib import Path

from .core import ANOMALOUS, NORMAL, DatasetIndex, Sample, discover_category, discover_split
from .integrity import LeakReport, audit
from .layout import LAYOUTS, MVTEC_AD, MVTEC_AD2, SYNTHETIC, DatasetLayout, get_layout
from .splits import SplitAssignment, carve_validation, ensure_validation
from .visa import VISA, VISA_CATEGORIES, VISA_GROUPS, discover_visa

__all__ = [
    "ANOMALOUS",
    "LAYOUTS",
    "MVTEC_AD",
    "MVTEC_AD2",
    "NORMAL",
    "SYNTHETIC",
    "VISA",
    "VISA_CATEGORIES",
    "VISA_GROUPS",
    "DatasetIndex",
    "DatasetLayout",
    "LeakReport",
    "Sample",
    "SplitAssignment",
    "audit",
    "carve_validation",
    "discover_category",
    "discover_split",
    "discover_visa",
    "ensure_validation",
    "get_layout",
    "load_category",
]


def load_category(
    root: str | Path,
    category: str,
    *,
    layout: str = "mvtec_ad2",
    split_csv: str | Path | None = None,
    require_masks: bool = True,
) -> dict[str, DatasetIndex]:
    """Index one category, whichever dataset it belongs to.

    VisA is driven by an official split CSV rather than by directory structure,
    so it needs a different reader; everything downstream receives the same
    `DatasetIndex` objects either way and does not need to know which dataset it
    is looking at.
    """
    if layout == "visa":
        return discover_visa(
            root, category, split_csv=split_csv, require_masks=require_masks
        )
    return discover_category(root, category, layout=layout, require_masks=require_masks)
