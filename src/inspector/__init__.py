"""MVTec Visual Inspector.

Industrial defect detection and localization trained on normal-only data.

Layering note: `inspector.data.core`, `inspector.metrics` and `inspector.fixtures`
are deliberately torch-free so the test suite runs in CI without a GPU or a
2.5 GB wheel. Torch appears only behind `inspector.models`, the service
(`inspector.api`) and the demo (`inspector.app`).
"""

__version__ = "1.0.1"
