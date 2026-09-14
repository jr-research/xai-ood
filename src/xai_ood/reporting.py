"""Reporting tiers: how much narrative each configuration earns in the report.

Declared 2026-08-31. Deciding after the numbers
are in which scorers deserve a paragraph and which get a line in an appendix is
a forking path: the tier assignment would then be a function of what won, and
the report would read as a narrowing of attention onto whatever happened to
look good.

**Tiering governs narrative, not computation.** All fifteen configurations run,
every one is scored on every protocol and every OOD group, and every one appears
in the results tables. A tier says how much prose a configuration earns,
nothing else. Nothing is thrown away, and promoting a configuration
later is a one-field change with an explicit record rather than a silent
reweighting of the story.

The tier *vocabulary* was fixed on 2026-08-31 and has not changed since. The
tier *assignment* has: two configurations were added on 2026-09-13, after
results existed, and were assigned ``appendix``. That amendment is dated in
three places that have to agree, which is the point of dating it: here, the
``methods.gaussian`` module docstring, and the transcribed map in
``tests/test_reporting_tiers.py``.

The four tiers
--------------
``primary``
    Carries the thesis's primary claims. The four 2x2 cells, because the
    factorial's two main effects and its interaction are computed from exactly
    those four; and ``knn_unnormalized``, because it is the non-parametric
    counterpoint the Gaussian family is measured against and the unnormalized
    variant is the primary one for the same reason the PCA-residual family's
    is.
``secondary``
    Reported with a paragraph, no primary claim beyond the two ablation
    contrasts already in the primary family. RMD and RMD++ (a decomposition the
    Full column already exposes), and the PCA-residual family's two unnormalized
    components.
``appendix``
    A table or a figure and a sentence. The normalization ablations that are not
    themselves a primary claim, and the shrinkage on/off ablation, which is a
    runtime flag rather than a configuration and so is tiered here by name
    (``SHRINKAGE_ABLATION_TIER``) rather than by a field. **Also the two
    normalised Gaussian Full cells added on 2026-09-13**, which are normalisation
    ablations by the same definition and are additionally post-hoc; this is the
    lowest tier that still guarantees a configuration reaches the tables, which
    is the reason they are here and not higher. See the ``methods.gaussian``
    module docstring.
``conditional``
    Runs unconditionally; is *narrated* only if a pre-declared trigger fires.
    One member: ``pca_residual_class_mean_whitened``, the pre-declared
    remedy that is written up only if the degeneracy control fires
    (``rho >= DEGENERACY_RHO_THRESHOLD`` against the centred norm). If rho does
    not fire, the whitened variant is a computed-but-unnarrated variant, and
    saying so is not the same as not having run it.

Why the field lives on the configuration
----------------------------------------
The alternative is a table in a separate document and nothing in the code. That
fails in the one way that matters: a scorer added later gets no tier, nobody
notices, and the omission is discovered when something asks which tier it is
in. ``tests/test_reporting_tiers.py`` asserts every configuration in every
family carries a tier, and the field is keyword-only with no default, so a new
configuration cannot be constructed without one.
"""

from __future__ import annotations

__all__ = [
    "REPORTING_TIERS",
    "SHRINKAGE_ABLATION_TIER",
    "validate_reporting_tier",
]

#: The four tiers, in descending narrative weight. ``conditional`` is last
#: because it is not a rank: it is a tier whose narration depends on a
#: pre-declared trigger rather than on a decision made now.
REPORTING_TIERS: tuple[str, ...] = ("primary", "secondary", "appendix", "conditional")

#: The shrinkage on/off ablation is a runtime flag on ``fit_covariance``, not a
#: configuration, so it has nowhere to carry a field. Its tier is declared here
#: so the declaration and the code agree about it, and so a reader looking
#: for "where is the shrinkage ablation tiered" finds an answer in the same
#: module as everything else.
SHRINKAGE_ABLATION_TIER: str = "appendix"


def validate_reporting_tier(tier: str, *, name: str) -> str:
    """Return ``tier`` if it is one of ``REPORTING_TIERS``, else raise.

    Called from every configuration's ``__post_init__``. A typo'd tier is a
    configuration that silently belongs to no tier at all, which is the failure
    the field exists to prevent, so it raises rather than warning.
    """
    if tier not in REPORTING_TIERS:
        raise ValueError(
            f"{name}: unknown reporting tier {tier!r}; the four declared tiers "
            f"are {list(REPORTING_TIERS)}. Tiers were fixed on 2026-08-31; "
            f"adding one is an amendment to the declared tier assignment, "
            f"not a code change."
        )
    return tier
