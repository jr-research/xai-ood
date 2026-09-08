# tests/test_crosscheck_availability.py
#
# Makes the differential cross-checks non-skippable by default.
#
# The problem this closes
# -----------------------
# `src/` is forbidden to import scipy and scikit-learn, and
# `tests/test_import_hygiene.py` enforces that. The consequence is that four
# things are reimplemented in numpy -- Ledoit-Wolf, Spearman, AUROC, and exact
# kNN search -- and the only evidence that any of them matches its reference is
# a differential test behind `pytest.importorskip`.
#
# `importorskip` is the right tool for the *test*: it un-skips wherever the
# library is present rather than turning the suite red where it is absent. But
# it makes the
# **suite** dishonest by default. In an environment without those libraries the
# eleven strongest tests in this project vanish and the summary line still reads
# green, so "all tests pass" would be true and would mean something much weaker
# than it appears to. That is exactly the failure mode this codebase spends its
# effort on everywhere else: a guard that does not fail on the bug it names.
#
# The fix is not to hard-import them, which would break the
# numpy-and-pandas-only property
# deliberately. It is to make their **absence** an explicit, acknowledged state
# rather than a silent one:
#
#   * By default, a missing cross-check library fails this file. Anyone running
#     the suite sees that the differential evidence was not collected.
#   * Setting `XAI_OOD_ALLOW_MISSING_CROSSCHECKS=1` turns that into a skip, for
#     the genuine numpy-and-pandas-only case. The
#     acknowledgement is then in the command line, where a reader of the run log
#     can see it, instead of nowhere.
#
# The packaging half of the same fix lives in `pyproject.toml`: see
# `REQUIRED_PYPROJECT_MARKERS` below. `pip install -e ".[test]"` should be all
# it takes to have the cross-checks run, so the default path is the honest one.
import ast
import os
import pathlib

import pytest

#: The environment variable that converts the failures below into skips. Named
#: rather than inlined so the message can quote it.
OPT_OUT = "XAI_OOD_ALLOW_MISSING_CROSSCHECKS"

#: What each library is the reference implementation *for*, so a failure says
#: which evidence is missing rather than only which import failed.
CROSSCHECK_LIBRARIES = {
    "scipy": "scipy.stats.spearmanr and rankdata, the reference for "
             "methods/correlation.py's numpy rank correlation",
    "sklearn": "sklearn.metrics.roc_auc_score and "
               "sklearn.covariance.ledoit_wolf_shrinkage, the references for "
               "metrics.auroc and covariance.ledoit_wolf_shrinkage",
}

#: Files carrying differential cross-checks, and how many `importorskip` calls
#: each holds. Pinned so that a cross-check deleted or accidentally rewritten
#: into an unconditional test is visible: the other way this guarantee erodes is
#: not the library going missing, it is the check going missing.
CROSSCHECK_FILES = {
    "test_covariance_estimator.py": 1,  # Ledoit-Wolf vs scikit-learn
    "test_metrics.py": 3,  # AUROC vs sklearn.metrics.roc_auc_score
    "test_spearman.py": 3,  # Spearman vs scipy.stats
    "test_independent_properties.py": 4,  # the differential probes
}

#: Strings `pyproject.toml` must contain for `pip install -e ".[test]"` to pull
#: the cross-check libraries. Asserted against the packaging file itself rather
#: than against prose describing it.
REQUIRED_PYPROJECT_MARKERS = (
    "[project.optional-dependencies]",
    "scipy",
    "scikit-learn",
)

TESTS_DIR = pathlib.Path(__file__).resolve().parent
PYPROJECT = TESTS_DIR.parent / "pyproject.toml"


def _opted_out() -> bool:
    return os.environ.get(OPT_OUT, "").strip().lower() in {"1", "true", "yes", "on"}


@pytest.mark.parametrize("library", sorted(CROSSCHECK_LIBRARIES))
def test_the_crosscheck_library_is_installed(library):
    """Fails, not skips, when a reference implementation is absent.

    This is the whole point of the file. Every other test that needs these
    libraries skips politely; this one does not, so the summary line cannot read
    green while the differential evidence is missing.
    """
    try:
        __import__(library)
    except ImportError:
        message = (
            f"{library} is not installed, so the differential cross-checks "
            f"against it did not run: {CROSSCHECK_LIBRARIES[library]}.\n"
            f"Install the test extras -- `pip install -e \".[test]\"` in the "
            f"repo -- or, if the omission is deliberate, set "
            f"{OPT_OUT}=1 so the omission is recorded in the command line "
            f"rather than nowhere."
        )
        if _opted_out():
            pytest.skip(message)
        pytest.fail(message)


def test_the_opt_out_is_off_unless_someone_set_it():
    """Reports the acknowledged opt-out state loudly rather than silently.

    A run with the opt-out set is a legitimate run. It is just not the run whose
    green summary means what a reader assumes, so it says so.
    """
    if _opted_out():
        pytest.skip(
            f"{OPT_OUT} is set: the differential cross-checks may not have run. "
            f"This suite's green line does not carry them."
        )


@pytest.mark.parametrize("filename,expected", sorted(CROSSCHECK_FILES.items()))
def test_each_crosscheck_file_still_carries_its_guarded_imports(filename, expected):
    """Pins the number of `importorskip` calls per file.

    Guards the other erosion path. If a cross-check is deleted, or rewritten to
    import its library unconditionally (which `test_import_hygiene.py` would
    catch) or to stop importing it at all (which nothing would), the count moves
    and this fails with the file named.
    """
    path = TESTS_DIR / filename
    assert path.exists(), f"{filename} is gone; it held {expected} cross-check(s)"
    tree = ast.parse(path.read_text())
    found = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "importorskip"
    )
    assert found == expected, (
        f"{filename} has {found} importorskip call(s), expected {expected}. A "
        f"differential cross-check was added or removed; update "
        f"CROSSCHECK_FILES deliberately and say so in the commit message."
    )


def test_the_test_extras_group_is_declared_in_packaging():
    """The packaging half of the same guarantee.

    Without an optional-dependency group naming them, whether the cross-checks
    run depends on whatever happens to be installed rather than on one documented
    install command. Asserting it against the packaging file means the group
    cannot be dropped without a test saying so.
    """
    if not PYPROJECT.exists():
        pytest.skip(
            f"no pyproject.toml beside tests/ at {PYPROJECT}; this check applies "
            f"to the packaged project, not to a bare source folder."
        )
    text = PYPROJECT.read_text()
    for marker in REQUIRED_PYPROJECT_MARKERS:
        assert marker in text, (
            f"pyproject.toml does not contain {marker!r}, so "
            f"`pip install -e \".[test]\"` will not pull the cross-check "
            f"libraries and whether they run depends on the environment."
        )
