# tests/test_reporting_tiers.py
#
# Guards the reporting-tier declarations, which were fixed on 2026-08-31.
#
# What is actually at risk here, and why prose alone would not have covered it:
#
#   * A scorer added later carries no tier, nobody notices, and the
#     omission surfaces later as "which tier is this in?" -- at
#     which point the answer is a function of how the scorer performed. The
#     keyword-only no-default field makes that a TypeError at construction; the
#     tests below make it a failure even if someone gives the field a default.
#   * The declared assignment drifts from the declaration it was transcribed
#     from. That declaration is the artefact an examiner reads; the code is
#     what ran. They have to agree, so the mapping is pinned here in full
#     rather than spot-checked.
#
# What this file deliberately does NOT assert: that a tier is *correct*. That is
# a judgement recorded in the declaration, not a property of the code.
import dataclasses

import pytest

from xai_ood.methods.gaussian import CONFIGURATIONS as GAUSSIAN_CONFIGURATIONS
from xai_ood.methods.gaussian import GaussianConfig
from xai_ood.methods.knn import CONFIGURATIONS as KNN_CONFIGURATIONS
from xai_ood.methods.knn import KnnConfig, fit_knn
from xai_ood.methods.pca_residual import CONFIGURATIONS as PCA_CONFIGURATIONS
from xai_ood.methods.pca_residual import PcaResidualConfig
from xai_ood.reporting import (
    REPORTING_TIERS,
    SHRINKAGE_ABLATION_TIER,
    validate_reporting_tier,
)

#: Every configuration the thesis ships, across all three families.
ALL_CONFIGURATIONS = (
    *GAUSSIAN_CONFIGURATIONS,
    *KNN_CONFIGURATIONS,
    *PCA_CONFIGURATIONS,
)

#: The declaration, transcribed. Written out in full rather than derived from
#: the code, so this compares the code against a statement of the same thing
#: that was written separately, and is not a tautology. It is a transcription
#: living in the same commit as the code, so it catches a drift between them
#: and cannot catch a change made deliberately to both at once.
DECLARED_TIERS = {
    "marginal_diagonal": "primary",
    "marginal_full": "primary",
    "class_conditional_diagonal": "primary",
    "class_conditional_full": "primary",
    "knn_unnormalized": "primary",
    "rmd": "secondary",
    "rmd_pp": "secondary",
    "pca_residual_all_id": "secondary",
    "pca_residual_class_mean": "secondary",
    "knn_normalized": "appendix",
    "pca_residual_all_id_l2": "appendix",
    "pca_residual_class_mean_l2": "appendix",
    "pca_residual_class_mean_whitened": "conditional",
    # AMENDMENT, 2026-09-13. Everything above this line is the 2026-08-31
    # declaration as transcribed. These two configurations did not exist then:
    # they were added after the first results existed, prompted by reading
    # Mueller and Hein 2025, and were assigned the lowest tier that still
    # guarantees a configuration reaches the results tables. The line is kept
    # here rather than tidied away because a reader checking the
    # pre-specification claim needs to see which entries postdate it, and a
    # sorted map with no seam cannot show that.
    "marginal_full_pp": "appendix",
    "class_conditional_full_pp": "appendix",
}


# --------------------------------------------------------------------------- #
# Every configuration carries a tier
# --------------------------------------------------------------------------- #


def test_there_are_fifteen_configurations_to_check():
    """Guards the guard: a parametrised absence check passes on an empty set.

    Thirteen until 2026-09-13, when the two post-hoc normalised Gaussian cells
    were added at tier ``appendix``.
    """
    assert len(ALL_CONFIGURATIONS) == 15, [c.name for c in ALL_CONFIGURATIONS]


def test_no_post_hoc_configuration_sits_above_the_appendix_tier():
    """The constraint the post-hoc addition is reported under, as an assertion.

    A cell added after the results may appear in the tables and may not earn
    more narrative weight than a cell fixed before them. ``primary`` and
    ``secondary`` are what "more narrative weight" means in this vocabulary, so
    naming the post-hoc set here and checking its tier is the only part of that
    constraint a test can carry. The rest lives in prose, where it belongs.
    """
    post_hoc = {"marginal_full_pp", "class_conditional_full_pp"}
    actual = {c.name: c.reporting_tier for c in ALL_CONFIGURATIONS}
    assert post_hoc <= set(actual), sorted(post_hoc - set(actual))
    for name in sorted(post_hoc):
        assert actual[name] == "appendix", (name, actual[name])


@pytest.mark.parametrize("config", ALL_CONFIGURATIONS, ids=lambda c: c.name)
def test_every_configuration_carries_a_declared_tier(config):
    assert config.reporting_tier in REPORTING_TIERS, (
        f"{config.name} carries tier {config.reporting_tier!r}, which is not one "
        f"of {list(REPORTING_TIERS)}"
    )


def test_the_tier_assignment_matches_the_declaration():
    """The code, pinned against the transcribed declaration above.

    Catches a tier changed in one place and not the other. It cannot catch a
    tier changed in both, which is why the pre-specification rests on the
    commit date rather than on this assertion."""
    actual = {c.name: c.reporting_tier for c in ALL_CONFIGURATIONS}
    assert actual == DECLARED_TIERS


def test_configuration_names_are_unique_across_families():
    """Two families sharing a name would collide in the tier map and in a dump."""
    names = [c.name for c in ALL_CONFIGURATIONS]
    assert len(set(names)) == len(names), sorted(names)


def test_every_tier_has_at_least_one_member():
    """A tier nobody is in is a tier that was declared and then forgotten."""
    used = {c.reporting_tier for c in ALL_CONFIGURATIONS}
    assert used == set(REPORTING_TIERS), sorted(set(REPORTING_TIERS) - used)


# --------------------------------------------------------------------------- #
# A later-added scorer cannot slip in untiered
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: GaussianConfig("x", False, True, False, False), id="gaussian"),
        pytest.param(lambda: KnnConfig("x", False), id="knn"),
        pytest.param(lambda: PcaResidualConfig("x", "all_id"), id="pca_residual"),
    ],
)
def test_a_configuration_cannot_be_built_without_a_tier(factory):
    """The field is keyword-only with no default, so this is a TypeError.

    This is the check that actually protects the declaration. A default
    would make an untiered scorer construct fine and land in whichever tier the
    default names, which is worse than no tier at all: it looks declared.
    """
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize(
    "cls,args",
    [
        (GaussianConfig, ("x", False, True, False, False)),
        (KnnConfig, ("x", False)),
        (PcaResidualConfig, ("x", "all_id")),
    ],
    ids=["gaussian", "knn", "pca_residual"],
)
def test_the_tier_field_is_keyword_only_with_no_default(cls, args):
    """Asserts the mechanism, not just its current effect.

    ``test_a_configuration_cannot_be_built_without_a_tier`` would also pass if
    the field were merely positional-and-required, which would break every
    existing call site instead of the intended one. This pins the shape.
    """
    fields = {f.name: f for f in dataclasses.fields(cls)}
    tier = fields["reporting_tier"]
    assert tier.kw_only is True
    assert tier.default is dataclasses.MISSING
    assert tier.default_factory is dataclasses.MISSING
    # And it really does construct with the keyword.
    assert cls(*args, reporting_tier="appendix").reporting_tier == "appendix"


@pytest.mark.parametrize(
    "cls,args",
    [
        (GaussianConfig, ("x", False, True, False, False)),
        (KnnConfig, ("x", False)),
        (PcaResidualConfig, ("x", "all_id")),
    ],
    ids=["gaussian", "knn", "pca_residual"],
)
def test_an_unknown_tier_is_rejected_at_construction(cls, args):
    with pytest.raises(ValueError, match="unknown reporting tier"):
        cls(*args, reporting_tier="headline")


def test_validate_reporting_tier_accepts_every_declared_tier():
    for tier in REPORTING_TIERS:
        assert validate_reporting_tier(tier, name="probe") == tier


# --------------------------------------------------------------------------- #
# The tier reaches the manifest
# --------------------------------------------------------------------------- #


def test_the_tier_survives_a_k_override():
    """``fit_knn(k=...)`` rebuilds the config; the tier must be carried, not defaulted.

    A sweep point is the same reported variant at another k, so silently
    retiering it there would make the appendix sweep and the headline table
    disagree about what ``knn_unnormalized`` is.
    """
    import numpy as np

    reference = np.arange(60, dtype=np.float64).reshape(20, 3)
    scorer = fit_knn("knn_unnormalized", reference, k=3)
    assert scorer.config.reporting_tier == "primary"
    assert scorer.diagnostics()["reporting_tier"] == "primary"


def test_gaussian_diagnostics_carry_the_tier():
    import numpy as np

    from xai_ood.methods.gaussian import fit_scorer

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    labels = np.tile(np.arange(4), 50)
    entry = fit_scorer("marginal_diagonal", x, labels).diagnostics()
    assert entry["reporting_tier"] == "primary"


def test_pca_residual_diagnostics_carry_the_tier():
    import numpy as np

    from xai_ood.methods.pca_residual import fit_pca_residual

    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 6))
    labels = np.tile(np.arange(4), 50)
    entry = fit_pca_residual("pca_residual_class_mean", x, labels).diagnostics()
    assert entry["reporting_tier"] == "secondary"


def test_the_tier_is_not_in_the_hyperparams_column():
    """A tier is a reporting decision, not a parameter of the estimator.

    The results table's ``hyperparams`` column is what distinguishes two runs of
    the same scorer. Promoting a configuration between tiers must not change it,
    or a promotion would look like a different scorer in every table already
    written.
    """
    from xai_ood.methods.gaussian import config_by_name as gaussian_config_by_name
    from xai_ood.methods.knn import config_by_name as knn_config_by_name
    from xai_ood.methods.pca_residual import config_by_name as pca_config_by_name
    from xai_ood.schema import decode_hyperparams

    assert "reporting_tier" not in decode_hyperparams(
        gaussian_config_by_name("rmd").encoded_hyperparams(shrinkage=False)
    )
    assert "reporting_tier" not in decode_hyperparams(
        knn_config_by_name("knn_normalized").encoded_hyperparams()
    )
    assert "reporting_tier" not in decode_hyperparams(
        pca_config_by_name("pca_residual_all_id").encoded_hyperparams(d=7)
    )


# --------------------------------------------------------------------------- #
# The one ablation that is not a configuration
# --------------------------------------------------------------------------- #


def test_the_shrinkage_ablation_is_tiered_by_name():
    """Shrinkage is a runtime flag on ``fit_covariance``, so it has no config.

    Declaring its tier as a module constant is what keeps the declared
    fifteen-configurations-plus-one-ablation list complete: without it the
    ablation would be the one reported thing with no tier anywhere.
    """
    assert SHRINKAGE_ABLATION_TIER in REPORTING_TIERS
    assert SHRINKAGE_ABLATION_TIER == "appendix"
