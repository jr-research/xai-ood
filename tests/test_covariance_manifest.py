# tests/test_covariance_manifest.py
#
# allclose(S, S.T) and the condition number must be logged for *every* fitted
# covariance, into the run manifest. That measurement is unavailable
# retroactively without refitting, and it is what converts the
# shrinkage-conditioning claim from an input-level argument into a result.
# These tests assert the fragment carries it.
import json

import numpy as np
import pytest

from xai_ood.methods.covariance import fit_covariance
from xai_ood.methods.gaussian import fit_all_scorers, fit_scorer
from xai_ood.methods.knn import fit_all_knn, fit_knn
from xai_ood.methods.pca_residual import degeneracy_report, fit_all_pca_residual
from xai_ood.methods.manifest import (
    SHRINKAGE_TARGET,
    covariance_manifest,
    covariance_manifest_entry,
    run_manifest,
    write_run_manifest,
)


def fixture(seed=1234, n=300):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=[-5.0, 0.0, 1.0], scale=[0.7, 0.5, 0.4], size=(n, 3))
    b = rng.normal(loc=[5.0, 0.0, -1.0], scale=[0.7, 0.5, 0.4], size=(n, 3))
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


def test_fragment_covers_every_covariance_including_rmd_backgrounds():
    x, labels = fixture()
    scorers = fit_all_scorers(x, labels)
    fragment = covariance_manifest(scorers)

    # Four plain cells contribute one covariance each; rmd and rmd_pp contribute
    # a primary and a background each. Eight in total.
    assert fragment["summary"]["n_covariances"] == 8
    assert "rmd:background" in fragment["covariances"]
    assert "rmd_pp:background" in fragment["covariances"]
    assert set(fragment["scorers"]) == set(scorers)


def test_every_entry_carries_symmetry_and_a_condition_number():
    x, labels = fixture()
    fragment = covariance_manifest(fit_all_scorers(x, labels))
    for name, entry in fragment["covariances"].items():
        assert entry["is_symmetric"] is True, name
        assert np.isfinite(entry["condition_number"]), name
        assert entry["condition_number"] >= 1.0, name


def test_condition_numbers_are_measured_per_covariance_not_shared():
    """The marginal and pooled within-class matrices are different matrices.

    The marginal one absorbs the between-class term, so on separated clusters
    it is markedly worse conditioned. If the fragment ever reported one number
    for both, the shrinkage ablation would have nothing to compare.
    """
    x, labels = fixture()
    fragment = covariance_manifest(fit_all_scorers(x, labels))
    conditions = fragment["summary"]["condition_numbers"]

    marginal = conditions["marginal_full:primary"]
    pooled = conditions["class_conditional_full:primary"]
    assert marginal != pooled
    assert marginal > pooled


def test_shrinkage_intensities_are_recorded_only_where_shrinkage_ran():
    x, labels = fixture()

    off = covariance_manifest(fit_all_scorers(x, labels, shrinkage=False))
    assert off["summary"]["shrinkage_intensities"] == {}

    on = covariance_manifest(fit_all_scorers(x, labels, shrinkage=True))
    assert len(on["summary"]["shrinkage_intensities"]) == 8
    for name, delta in on["summary"]["shrinkage_intensities"].items():
        assert 0.0 <= delta <= 1.0, name
        assert on["covariances"][name]["shrinkage_target"] == SHRINKAGE_TARGET


def test_the_shrinkage_target_is_named_rather_than_left_as_ledoit_wolf():
    """Three papers share the name and differ in their target.

    scaled identity (JMVA 2004, the one used), constant correlation (JPM 2004),
    single factor (JEF 2003). The manifest names the target so it is not left
    to be inferred.
    """
    assert "jmva" in SHRINKAGE_TARGET
    assert "scaled_identity" in SHRINKAGE_TARGET


def test_a_missing_diagnostic_fails_loudly_rather_than_writing_a_gap():
    x, labels = fixture()
    scorer = fit_scorer("marginal_full", x)
    del scorer.primary.diagnostics["condition_number"]
    with pytest.raises(KeyError, match="condition_number"):
        covariance_manifest_entry(scorer.primary)


def test_fragment_is_json_serialisable_and_carries_the_run_level_provenance(tmp_path):
    """Both commit hashes and the seed belong at the top level of the manifest.

    This function does not invent them -- a manifest recording a guessed hash is
    worse than one recording none -- but it cannot write one without them
    either.
    """
    x, labels = fixture()
    scorers = fit_all_scorers(x, labels, shrinkage=True)
    path = tmp_path / "runs" / "run_metadata.json"

    returned = write_run_manifest(
        path,
        scorers,
        seed=20260825,
        repo_commit="0" * 40,
        openood_commit="8d44375e4c695d03d2b97850b754f24fd4bda447",
    )

    on_disk = json.loads(path.read_text())
    assert on_disk == returned
    assert on_disk["openood_commit"] == "8d44375e4c695d03d2b97850b754f24fd4bda447"
    assert on_disk["repo_commit"] == "0" * 40
    assert on_disk["seed"] == 20260825
    assert on_disk["covariance_summary"]["all_symmetric"] is True
    assert on_disk["covariance_summary"]["condition_number_max"] == max(
        on_disk["covariance_summary"]["condition_numbers"].values()
    )


# --------------------------------------------------------------------------- #
# Cross-family: what a full run actually writes
# --------------------------------------------------------------------------- #


def test_covariance_manifest_skips_scorers_that_have_no_covariance():
    """A full run hands the manifest every scorer, not just the Gaussians.

    kNN fits no covariance, and the PCA-residual scorer fits a subspace
    rather than one. Before this was handled, passing a mixed collection raised
    `AttributeError: 'KnnScorer' object has no attribute 'primary'` -- a crash at
    the end of a multi-hour run, which is the worst possible time.
    """
    x, labels = fixture()
    mixed = {**fit_all_scorers(x, labels), **fit_all_knn(x, k=5)}
    fragment = covariance_manifest(mixed)
    assert fragment["summary"]["n_covariances"] == 8  # the Gaussian ones only
    assert not any("knn" in k for k in fragment["covariances"])


def test_run_manifest_covers_every_family_through_their_own_diagnostics():
    """A new family joins the manifest by implementing `diagnostics()`, not by
    editing the manifest module."""
    x, labels = fixture()
    scorers = {**fit_all_scorers(x, labels, shrinkage=True), **fit_all_knn(x, k=5)}
    fragment = run_manifest(scorers, seed=20260825, repo_commit="0" * 40, openood_commit="8d44375e")

    assert fragment["n_scorers"] == 8
    assert set(fragment["scorers"]) == set(scorers)
    assert fragment["scorers"]["knn_normalized"]["n_reference"] == len(x)
    assert fragment["scorers"]["knn_normalized"]["k"] == 5
    assert fragment["covariance_summary"]["n_covariances"] == 8


def test_run_manifest_omits_the_covariance_section_when_there_is_none():
    fragment = run_manifest(fit_all_knn(fixture()[0], k=5), seed=20260825, repo_commit="0" * 40, openood_commit="8d44375e")
    assert "covariances" not in fragment
    assert fragment["n_scorers"] == 2


def test_run_manifest_refuses_duplicate_scorer_names():
    """Same failure the score dump guards: one entry silently overwriting another."""
    x, _ = fixture()
    a = fit_knn("knn_normalized", x, k=5)
    b = fit_knn("knn_normalized", x, k=9)
    with pytest.raises(ValueError, match="duplicate scorer name"):
        run_manifest([a, b], seed=20260825, repo_commit="0" * 40, openood_commit="8d44375e")


def test_write_run_manifest_is_json_and_carries_both_commit_hashes(tmp_path):
    x, labels = fixture()
    scorers = {**fit_all_scorers(x, labels), **fit_all_knn(x, k=5)}
    path = tmp_path / "runs" / "run_metadata.json"
    returned = write_run_manifest(
        path,
        scorers,
        seed=20260825,
        repo_commit="0" * 40,
        openood_commit="8d44375e4c695d03d2b97850b754f24fd4bda447",
    )
    assert json.loads(path.read_text()) == returned
    assert returned["openood_commit"] == "8d44375e4c695d03d2b97850b754f24fd4bda447"
    assert returned["repo_commit"] == "0" * 40
    assert returned["seed"] == 20260825


def test_no_diagnostic_is_a_hardcoded_constant():
    """The three runtime-measured quantities must be read at runtime, always.

    Condition numbers are the one of the three this module owns. Refitting the
    same scorer on different data must move the number; if it did not, it would
    be a constant somewhere.
    """
    x_a, labels_a = fixture(seed=1)
    x_b, labels_b = fixture(seed=2, n=40)

    a = covariance_manifest(fit_all_scorers(x_a, labels_a))["summary"]["condition_numbers"]
    b = covariance_manifest(fit_all_scorers(x_b, labels_b))["summary"]["condition_numbers"]

    assert set(a) == set(b)
    assert all(a[k] != b[k] for k in a)


# --------------------------------------------------------------------------- #
# Provenance: two commit hashes in every manifest
# --------------------------------------------------------------------------- #


def test_run_manifest_cannot_be_called_without_seed_and_both_commit_hashes():
    """The seed and both commit hashes are required in *every* manifest.

    They are required keyword arguments rather than validated keys, so omitting
    one is a TypeError at the call site -- unforgeable, and impossible to reach
    the disk without. A validated `extra` dict would still let a script write a
    provenance-free manifest that looks complete; this is the failure that costs
    a refit of every covariance in the study, and a scoring run is the one stage
    that cannot re-derive the values afterwards.

    Mutation killed: dropping any of the three parameters back to a default, or
    demoting them to a checked key in `extra`, makes this test pass a call that
    should not be expressible.
    """
    x, labels = fixture()
    scorers = fit_all_scorers(x, labels)

    with pytest.raises(TypeError):
        run_manifest(scorers)
    with pytest.raises(TypeError):
        run_manifest(scorers, seed=1, repo_commit="a" * 40)
    with pytest.raises(TypeError):
        run_manifest(scorers, seed=1, openood_commit="b" * 40)


def test_a_scorer_declaring_a_required_key_is_refused_without_it():
    """A scorer can declare a quantity the manifest must carry, and be enforced.

    Rho is a worse loss than a missing commit hash. A missing hash is
    embarrassing; a missing rho means the **pre-declared threshold cannot be
    applied to the run that actually happened**, which dissolves the
    falsification claim the whole degeneracy control exists to support.
    Recomputing it afterwards, against a possibly different scored set, is the
    forking path the pre-declaration closes. And `run_manifest(scorers, ...)`
    takes scorers only, while `degeneracy_report(scorer, embeddings)` needs the
    scored rows too, so there is no way for the manifest to compute it itself.

    The dependency is inverted rather than special-cased: `manifest.py` knows
    nothing about the PCA-residual family. The scorer declares
    `MANIFEST_REQUIRES` and
    `run_manifest` enforces whatever is declared, the same way a new family
    joins the manifest by implementing `diagnostics()` rather than by editing
    this function. A later scorer with its own unrecoverable quantity gets the
    same guard for free.
    """
    x, labels = fixture()
    scorers = fit_all_pca_residual(x, labels)

    # Every PCA-residual scorer declares it, so the requirement cannot be dodged by
    # fitting a subset.
    for scorer in scorers.values():
        assert "degeneracy" in scorer.MANIFEST_REQUIRES

    with pytest.raises(ValueError, match="degeneracy"):
        run_manifest(
            scorers, seed=20260826, repo_commit="a" * 40, openood_commit="8d44375e"
        )

    # Supplying it satisfies the guard, and it lands in the manifest.
    report = degeneracy_report(scorers["pca_residual_class_mean"], x)
    fragment = run_manifest(
        scorers,
        seed=20260826,
        repo_commit="a" * 40,
        openood_commit="8d44375e",
        extra={"degeneracy": report},
    )
    assert fragment["degeneracy"]["rho_centred_norm"] == report["rho_centred_norm"]
    assert fragment["degeneracy"]["threshold_applies_to"] == "rho_centred_norm"


def test_scorers_declaring_nothing_are_unaffected():
    """The guard is opt-in: families that declare nothing keep working."""
    x, labels = fixture()
    for scorer in fit_all_scorers(x, labels).values():
        assert getattr(scorer, "MANIFEST_REQUIRES", ()) == ()
    fragment = run_manifest(
        fit_all_scorers(x, labels),
        seed=20260826,
        repo_commit="a" * 40,
        openood_commit="8d44375e",
    )
    assert fragment["n_scorers"] > 0


def test_run_manifest_records_the_provenance_it_was_given():
    x, labels = fixture()
    scorers = fit_all_scorers(x, labels)
    fragment = run_manifest(
        scorers, seed=20260826, repo_commit="a" * 40, openood_commit="8d44375e"
    )
    assert fragment["seed"] == 20260826
    assert fragment["repo_commit"] == "a" * 40
    assert fragment["openood_commit"] == "8d44375e"


def test_covariance_manifest_tests_the_type_not_the_attribute_name():
    """A scorer that names its fitted subspace `primary` must not crash.

    Filtering on `hasattr(s, "primary")` is not enough: `primary` is exactly what
    a PCA-residual scorer would plausibly call its fitted object. That family has
    two subspaces, a natural primary/secondary pair, and `GaussianScorer` sets
    the naming precedent. Such a scorer passes the filter and then hits
    `scorer.background` on the next line, failing at the worst moment: the end of
    a multi-hour scoring run, with `AttributeError: 'PcaResidualScorer' object
    has no attribute 'background'`.

    Mutation killed: reverting the filter to `hasattr(s, "primary")`.
    """

    class PcaResidualScorer:
        """Shaped like what the PCA-residual scorer writes."""

        name = "pca_residual_all_id"

        def __init__(self):
            self.primary = object()  # a fitted subspace, not a CovarianceFit

        def diagnostics(self):
            return {"scorer": self.name, "d": 41}

    x, labels = fixture()
    mixed = [*fit_all_scorers(x, labels).values(), PcaResidualScorer()]

    fragment = covariance_manifest(mixed)
    assert fragment["summary"]["n_covariances"] == 8  # the Gaussian ones only
    assert not any("pca" in k for k in fragment["covariances"])

    run = run_manifest(
        mixed, seed=1, repo_commit="a" * 40, openood_commit="8d44375e"
    )
    assert run["n_scorers"] == 7
    assert run["scorers"]["pca_residual_all_id"]["d"] == 41


def test_covariance_manifest_refuses_two_fits_sharing_a_name():
    """`covariances[fit.name] = ...` silently drops the first of a colliding pair.

    `run_manifest` guards duplicate *scorer* names and raises; nothing guarded
    duplicate *fit* names. Verified before the fix: two genuinely different
    CovarianceFits both named "shared_name" produced `n_covariances = 1`, the
    first silently gone, no error. Unreachable through `fit_scorer`, which
    derives names from config names, reachable the moment the PCA-residual
    family or a full run constructs fits by hand.

    Mutation killed: removing the collision check restores the silent drop.
    """
    x, _ = fixture()
    first = fit_covariance(x, name="shared_name")
    second = fit_covariance(x * 2.0, name="shared_name")

    class OneFit:
        def __init__(self, name, fit):
            self.name = name
            self.primary = fit
            self.background = None
            self.shrinkage = False
            self.config = _StubConfig()

        def diagnostics(self):
            return {}

    with pytest.raises(ValueError, match="two covariance fits share the name"):
        covariance_manifest([OneFit("a", first), OneFit("b", second)])


class _StubConfig:
    def encoded_hyperparams(self, **_kwargs):
        return "{}"


def test_manifest_reports_how_many_covariances_are_actually_distinct():
    """Eight entries, six distinct matrices, and the manifest says both.

    `rmd:primary` is bit-identical to `class_conditional_full:primary`, and
    `rmd:background` to `marginal_full:primary` -- RMD's two terms *are* the two
    cells of the Full column, which is the decomposition the thesis reports, so
    fitting them twice is deliberate and the fits stay independent. But a reader
    of `condition_numbers` sees the same number twice under two names and cannot
    tell that from two genuinely different matrices that happen to agree.

    Eight is still what gets reported: each entry records what a scorer actually
    used. `n_distinct_covariances` says how many underlying matrices that is.

    Mutation killed: reporting `n_distinct_covariances == n_covariances`.
    """
    x, labels = fixture()
    fragment = covariance_manifest(fit_all_scorers(x, labels))
    summary = fragment["summary"]

    assert summary["n_covariances"] == 8
    assert summary["n_distinct_covariances"] == 6

    covs = {
        name: entry
        for name, entry in fragment["covariances"].items()
    }
    assert len(covs) == 8


def test_there_is_exactly_one_manifest_writer():
    """`write_covariance_manifest` is retired; `write_run_manifest` is the writer.

    Two writers where one enforces provenance and one does not is the same
    "two implementations of one rule" smell this codebase avoids everywhere
    else -- one `l2_normalize`, one `score_frame`, one covariance estimator. A
    safe path and an unsafe path side by side means the unsafe one eventually
    gets called, and the failure is a file on disk that looks complete, cannot
    be defended later, and costs a refit to regenerate.

    `covariance_manifest()` -- the pure function -- stays: it is what
    `run_manifest` calls, and anything that genuinely needs a covariance-only
    fragment can call it and write the file itself, deliberately.

    Mutation killed: re-exporting `write_covariance_manifest` from either
    module.
    """
    import xai_ood.methods as methods
    from xai_ood.methods import manifest as manifest_module

    for module in (manifest_module, methods):
        assert not hasattr(module, "write_covariance_manifest"), (
            f"{module.__name__} still exposes a manifest writer that does not "
            f"require the seed and both commit hashes"
        )
        assert "write_covariance_manifest" not in module.__all__

    # The pure function stays, and stays JSON-serialisable on its own.
    x, labels = fixture()
    fragment = covariance_manifest(fit_all_scorers(x, labels))
    assert json.loads(json.dumps(fragment)) == fragment
