"""Plotting for the thesis. The style is locked in ``xai_ood.visualization.style``;
import from there rather than setting rcParams anywhere else.
"""

from xai_ood.visualization.style import (  # noqa: F401
    AUROC_LABEL,
    AUROC_SCALE,
    FPR95_LABEL,
    OKABE_ITO,
    SEQUENTIAL_CMAP,
    apply_style,
    format_auroc,
    new_figure,
    save_figure,
    series_color,
    severity_colors,
    to_display_scale,
)
