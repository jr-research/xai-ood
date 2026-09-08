# tests/test_import_hygiene.py
#
# Guards a load-bearing property that is otherwise
# invisible: **the scorer package must import nothing beyond numpy and pandas.**
#
# This is not tidiness. Three implementation decisions exist only because of it:
#
#   * Ledoit-Wolf is reimplemented in numpy rather than importing
#     sklearn.covariance, so the covariance tests need no optional dependency.
#   * the Gaussian family uses np.linalg.solve on the Cholesky factor rather than
#     scipy.linalg.solve_triangular.
#   * kNN uses numpy exact search rather than faiss, which would otherwise have
#     been the obvious call.
#
# Each of those cost something. They are worth it only as long as the property
# holds, and the property is easy to break by accident: the moment scipy or
# scikit-learn is installed in a development environment -- and someone may well
# install them precisely to cross-check this code -- a stray `import scipy`
# will work fine locally and silently make the whole suite
# undevelopable on a machine without it. Nothing else would catch that.
#
# scipy, scikit-learn and faiss are still perfectly fine as **verification
# tools**, called from a test behind `pytest.importorskip`. The rule is only
# that `src/` must not depend on them, and that a test must skip rather than
# fail when they are absent. Both halves are checked below.
import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "xai_ood"

#: Everything the scorer package may import at module level. Adding to this list
#: is a real decision, not a formality: it changes what someone needs installed
#: to develop and test this package.
ALLOWED_TOP_LEVEL = {
    "__future__",
    "numpy",
    "pandas",
    "ast",
    "dataclasses",
    "json",
    "pathlib",
    # Added deliberately, per this list's own "adding to this list is a real
    # decision" rule. `csid_imglist.py` records the
    # SHA-256 of the source ID-test imglist and of the emitted one, because the
    # 2,000-image draw is a function of the seed *and* of the source file's row
    # order -- a seed alone does not reproduce it. hashlib is stdlib, so it
    # changes nothing about what has to be installed to develop or test this
    # package, which is the property this list exists to protect.
    #
    # Not a precedent for `methods/`: `manifest.py` deliberately used builtin
    # `hash` rather than hashlib for its fit digests, and that reasoning
    # (per-process salt is fine for a within-call count) still stands. This is
    # the different case -- a digest written to disk and compared across
    # machines and months, which builtin `hash` cannot do.
    "hashlib",
    "typing",
    "collections",
    "itertools",
    "math",
    "functools",
    "warnings",
}

#: Importable inside a function body, never at module level. matplotlib is
#: heavy and pulls a backend; `style.apply_style()` imports it lazily so the
#: module stays importable (and testable) without it.
ALLOWED_LAZY = {"matplotlib", "cycler", "mpl_toolkits"}

FORBIDDEN = {"scipy", "sklearn", "faiss", "torch", "openood"}


def source_files():
    return sorted(p for p in SRC.rglob("*.py"))


def module_level_imports(path: pathlib.Path) -> set[str]:
    """Top-level import names only. Imports nested in a function are lazy."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:  # module level only, deliberately not ast.walk
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import within xai_ood
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_there_are_source_files_to_check():
    """Guards against the guard silently passing on an empty directory."""
    files = source_files()
    assert len(files) >= 5, f"only found {files}"


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_no_module_level_import_outside_the_allowed_set(path):
    offenders = module_level_imports(path) - ALLOWED_TOP_LEVEL - {"xai_ood"}
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)} at module level. The scorer "
        f"package must stay importable with only numpy and pandas installed -- "
        f"that is why Ledoit-Wolf is reimplemented and faiss is not used. If "
        f"this dependency is genuinely necessary, add it to ALLOWED_TOP_LEVEL "
        f"deliberately and say so in the commit message; do not widen the set "
        f"to make a test pass."
    )


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_no_forbidden_dependency_anywhere_in_the_package(path):
    """Stricter than the above: scipy and friends are out even lazily.

    A lazy `import scipy` inside a function would pass the module-level check
    and then fail at runtime on a machine without it -- the worst version, since
    it surfaces during a run rather than at import.
    """
    text = path.read_text()
    tree = ast.parse(text)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module.split(".")[0])
    offenders = found & FORBIDDEN
    assert not offenders, (
        f"{path.name} imports {sorted(offenders)}. These may be used in tests "
        f"as verification tools behind pytest.importorskip, but never in src/."
    )


def test_lazy_imports_are_confined_to_function_bodies():
    """matplotlib is allowed, but only lazily. Asserts the split is real."""
    for path in source_files():
        at_module_level = module_level_imports(path) & ALLOWED_LAZY
        assert not at_module_level, (
            f"{path.name} imports {sorted(at_module_level)} at module level; "
            f"import it inside the function that needs it instead, so the "
            f"module stays importable without a plotting backend."
        )


def test_optional_dependencies_are_reached_only_through_importorskip():
    """A test that hard-imports scipy would fail, not skip, where it is absent.

    The distinction matters: `pytest.importorskip` turns an absent optional
    dependency into a skip that un-skips automatically where it is present,
    which is exactly how the Ledoit-Wolf cross-check is meant to behave. A bare
    `import sklearn` at the top of a test file would turn the entire suite red
    wherever it is absent.
    """
    tests_dir = pathlib.Path(__file__).resolve().parent
    for path in sorted(tests_dir.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            names: set[str] = set()
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = {node.module.split(".")[0]}
            offenders = names & FORBIDDEN
            assert not offenders, (
                f"{path.name} imports {sorted(offenders)} at module level. Use "
                f"pytest.importorskip inside the test that needs it, so the "
                f"suite skips rather than fails when it is absent."
            )


#: Numerical routines this package rules out, mapped to the reason. All four
#: are attributes of `numpy.linalg`, so no import guard catches them -- only an
#: AST scan for the attribute access does.
FORBIDDEN_NUMPY_LINALG = {
    "eig": "Use eigh, never eig. On a symmetric matrix eig can return "
           "complex eigenpairs from numerical asymmetry. Use eigh/eigvalsh.",
    "eigvals": "Use eigh, never eig. Use eigvalsh.",
    "inv": "Prefer a Cholesky solve over forming an explicit inverse.",
    "pinv": "Do not silently pseudo-invert. A singular covariance is "
            "a condition to raise on, not to paper over.",
}


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_no_forbidden_numpy_linalg_routine_in_the_package(path):
    """The eigh and inverse rules, as a guard rather than as inspection.

    The code is correct -- `eigvalsh` in `_covariance_diagnostics`, a cached
    Cholesky factor in `mahalanobis_sq`, no inverse anywhere -- but it was
    correct *by inspection only*: mutating `eigvalsh` to `eig`, and the Cholesky
    solve to an explicit `np.linalg.inv`, left the entire suite passing both
    times.

    Why the existing tests could not catch it:

      * `test_condition_number_is_real_and_matches_eigh` computes
        `np.linalg.eigvalsh` **in the test** and compares the value. On a
        symmetric matrix `eig` and `eigh` agree to ~1e-14, well inside its
        tolerance, and `assert np.isrealobj(...)` is applied to the test's own
        call rather than the module's. It checks a number, not a routine.
      * `test_mahalanobis_matches_explicit_inverse_without_forming_one`
        *asserts the two agree*, which is precisely what it cannot use to tell
        them apart. The name over-claims.

    A value-comparing test structurally cannot distinguish these; the choice of
    routine is the thing being required, so the routine is what gets asserted.
    The same requirement covers the PCA eigendecomposition.

    Mutations killed: `eigvalsh` -> `np.real(np.linalg.eigvals(...))`, and the
    Cholesky solve -> `np.linalg.inv(...)`.

    Note what this does *not* cover: the `0.5 * (S + S.T)` symmetrisation.
    `R.T @ R` is bitwise symmetric on every BLAS and shape measured, so
    deleting the line changes no observable value and no behavioural test can
    fail. It is guarded by inspection rather than by a test, and that is stated
    rather than implied.
    """
    tree = ast.parse(path.read_text())
    offenders = {}
    for node in ast.walk(tree):
        # np.linalg.eig(...) / numpy.linalg.inv(...) -- an Attribute whose own
        # value is an Attribute named `linalg`.
        if (
            isinstance(node, ast.Attribute)
            and node.attr in FORBIDDEN_NUMPY_LINALG
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "linalg"
        ):
            offenders[node.attr] = FORBIDDEN_NUMPY_LINALG[node.attr]
    assert not offenders, (
        f"{path.name} uses np.linalg.{sorted(offenders)}:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in sorted(offenders.items()))
    )


def test_the_forbidden_linalg_scan_actually_matches_something():
    """Guards the guard: an AST pattern that matches nothing passes vacuously.

    The check above asserts an absence, so a typo in the matcher would make it
    permanently green. This feeds it source that *should* trip it.
    """
    tree = ast.parse("import numpy as np\nx = np.linalg.eig(a)\ny = np.linalg.inv(b)\n")
    hits = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in FORBIDDEN_NUMPY_LINALG
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "linalg"
    }
    assert hits == {"eig", "inv"}
