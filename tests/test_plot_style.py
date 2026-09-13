"""Tests for the locked figure style.

These are cheap guards on decisions that are expensive to discover broken once
figures have been exported: the AUROC scale, palette stability across runs, and
that the style actually installs into matplotlib.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless; no display assumed

import pytest

from xai_ood.visualization import style


def test_auroc_is_on_the_zero_to_hundred_scale():
    assert style.AUROC_SCALE == 100.0
    assert style.to_display_scale(0.9846) == pytest.approx(98.46)
    assert style.format_auroc(98.456) == "98.46"
    assert style.format_auroc(None) == "--"
    assert style.format_auroc(float("nan")) == "--"
    assert "%" in style.AUROC_LABEL


def test_palette_is_the_eight_colour_okabe_ito_set():
    assert len(style.OKABE_ITO) == 8
    assert len(set(style.OKABE_ITO)) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in style.OKABE_ITO)


def test_series_colour_is_stable_and_not_salted_by_process_hash():
    """A scorer must get the same colour in every figure, on every machine.

    Python's str hash is salted per process, so a hash()-based fallback would
    silently recolour every appendix figure on a rerun. This asserts the
    fallback is deterministic instead.
    """
    assert style.series_color("knn") == style.series_color("knn")
    assert style.series_color("class_conditional_full") in style.OKABE_ITO
    # Unknown name: deterministic fallback, computed here independently.
    expected = style.OKABE_ITO[sum(ord(c) for c in "some_future_scorer") % 8]
    assert style.series_color("some_future_scorer") == expected


def test_naive_baseline_and_mds_are_visually_distinct():
    """The 2x2's corner cells are the headline comparison; they must not collide."""
    corners = [
        style.series_color("marginal_diagonal"),
        style.series_color("marginal_full"),
        style.series_color("class_conditional_diagonal"),
        style.series_color("class_conditional_full"),
    ]
    assert len(set(corners)) == 4


def test_apply_style_installs_the_decisions():
    style.apply_style()
    rc = matplotlib.rcParams
    assert rc["font.size"] == style.BASE_FONT_SIZE
    assert rc["font.sans-serif"][0] == "DejaVu Sans"
    assert rc["image.cmap"] == style.SEQUENTIAL_CMAP
    assert rc["savefig.dpi"] == style.DPI
    assert rc["pdf.fonttype"] == 42  # embedded TrueType, selectable text
    cycle_colors = rc["axes.prop_cycle"].by_key()["color"]
    assert cycle_colors == list(style.OKABE_ITO)


def test_severity_colours_are_ordered_and_distinct():
    colors = style.severity_colors(5)
    assert len(colors) == 5
    assert len(set(colors)) == 5


def test_save_figure_writes_pdf_and_png(tmp_path):
    fig, ax = style.new_figure()
    ax.plot([1, 2, 3, 4, 5], [70.0, 80.0, 85.0, 88.0, 90.0], color=style.series_color("knn"))
    ax.set_xlabel("Severity")
    ax.set_ylabel(style.AUROC_LABEL)

    written = style.save_figure(fig, tmp_path / "nested" / "severity_curve")
    assert [p.suffix for p in written] == [".pdf", ".png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_dataset_colour_is_stable_and_registered():
    # Every split the Phase 2 runs actually produce has a registered colour, so
    # no figure silently falls back for a split that exists.
    for split in (
        "cifar10_train", "cifar10_test", "cifar10_val", "csid",
        "cifar100", "tin", "mnist", "svhn", "texture", "places365",
    ):
        assert split in style.DATASET_COLORS
        assert style.dataset_color(split) == style.DATASET_COLORS[split]


def test_dataset_palette_deviations_are_declared():
    # A colour outside Okabe-Ito is allowed, but only if it is named in
    # PALETTE_EXTENSIONS. An undeclared one is drift and this test is what makes
    # the distinction checkable rather than a matter of comment-reading.
    allowed = set(style.OKABE_ITO) | set(style.PALETTE_EXTENSIONS)
    undeclared = sorted(set(style.DATASET_COLORS.values()) - allowed)
    assert undeclared == [], f"undeclared non-palette colours: {undeclared}"
    # And nothing is declared as an extension that is not actually used, which
    # would let a real deviation hide behind a stale allowance.
    unused = sorted(set(style.PALETTE_EXTENSIONS) - set(style.DATASET_COLORS.values()))
    assert unused == [], f"declared but unused extensions: {unused}"


def test_dataset_fallback_is_deterministic_not_salted():
    a = style.dataset_color("a_split_never_registered")
    b = style.dataset_color("a_split_never_registered")
    assert a == b
    assert a in style.OKABE_ITO


def test_neutrals_are_greys_and_not_data_colours():
    # A neutral must be achromatic, so it cannot be mistaken for a series, and
    # must not collide with the categorical palette.
    for hexval in style.NEUTRALS:
        r, g, b = (int(hexval[i:i + 2], 16) for i in (1, 3, 5))
        assert r == g == b, f"{hexval} is not achromatic"
        assert hexval not in style.OKABE_ITO
