"""Run-manifest fragment for the covariance fits.

Every fitted covariance must have ``np.allclose(S, S.T)`` and its condition
number logged. This module turns the diagnostics that
``xai_ood.methods.covariance.fit_covariance`` already collects into a
JSON-serialisable fragment that a run's ``run_metadata.json`` can absorb.

Why this is a module and not three lines at a call site: the measurement is
**unavailable retroactively without refitting**, and it is the thing that
settles the shrinkage-conditioning claim. That claim is an input-level
argument (~65 samples per dimension on these cells, so the sample
covariance *should* be well conditioned) and not a result. Logging the numbers
is what converts it. Losing it because a script forgot to write it out would
cost a refit of every covariance in the study.

**Every value in this fragment is read at runtime.** The three quantities that
can only be measured once the embeddings exist -- the PCA-residual degeneracy
rho, the per-covariance condition numbers, and the integer d from the
95%-variance rule -- must never appear as constants in a source file.
Condition numbers are the one of the three this module owns; it computes them,
it does not know them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .covariance import CovarianceFit

__all__ = [
    "covariance_manifest_entry",
    "covariance_manifest",
    "run_manifest",
    "write_run_manifest",
    "SHRINKAGE_TARGET",
]

#: Named rather than written as "Ledoit-Wolf shrinkage" unqualified, because
#: three papers share the name and differ in their target: scaled identity
#: (JMVA 2004, this one), constant correlation (JPM 2004), single factor
#: (JEF 2003). Recorded in every manifest so the artefact names the target
#: rather than leaving it to be inferred.
SHRINKAGE_TARGET: str = "ledoit_wolf_2004_jmva_scaled_identity"

#: The diagnostics every covariance manifest must carry, checked on write so a
#: refactor of the diagnostics dict cannot quietly drop one.
_REQUIRED_KEYS: tuple[str, ...] = (
    "is_symmetric",
    "raw_is_symmetric",
    "raw_max_asymmetry",
    "condition_number",
    "shrinkage",
    "shrinkage_intensity",
    "eigenvalue_min",
    "eigenvalue_max",
    "n_samples",
    "n_features",
    "samples_per_dimension",
)


def covariance_manifest_entry(fit: CovarianceFit) -> dict[str, Any]:
    """One covariance's manifest entry, JSON-serialisable."""
    entry = dict(fit.diagnostics)
    missing = [k for k in _REQUIRED_KEYS if k not in entry]
    if missing:
        raise KeyError(
            f"{fit.name}: diagnostics are missing {missing}. Every fitted "
            f"covariance must log symmetry and its condition number, and neither "
            f"is recoverable after the run without refitting."
        )
    if entry["shrinkage"]:
        entry["shrinkage_target"] = SHRINKAGE_TARGET
    return entry


def _as_list(scorers: Mapping[str, Any] | Iterable[Any]) -> list[Any]:
    return list(scorers.values()) if isinstance(scorers, Mapping) else list(scorers)


def _fit_digest(fit: CovarianceFit) -> tuple[Any, ...]:
    """Content key for a fit's mean and covariance, for counting distinct ones.

    Bitwise, not tolerant: two fits are "the same matrix" here only if they are
    the same matrix. The pairs this actually catches (``rmd:primary`` against
    ``class_conditional_full:primary``) are byte-identical because they come
    from the same estimator on the same input, not merely close.

    Uses the builtin ``hash`` of the raw bytes rather than ``hashlib``, so this
    module does not widen ``test_import_hygiene.ALLOWED_TOP_LEVEL`` for a
    counting convenience. The value is never written to the manifest and never
    compared across processes -- only within a single call -- so ``hash``'s
    per-process salt is irrelevant, and the shapes are carried alongside so a
    reshape cannot alias.
    """
    parts: list[Any] = []
    for array in (fit.mean, fit.covariance):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        parts.append((contiguous.shape, hash(contiguous.tobytes())))
    return tuple(parts)


def covariance_manifest(
    scorers: Mapping[str, Any] | Iterable[Any],
) -> dict[str, Any]:
    """Manifest fragment covering every covariance behind every scorer.

    **Keyed by fit name, one entry per covariance a scorer actually used.**
    Names are ``f"{config.name}:primary"`` and ``f"{config.name}:background"``,
    so they are unique by construction and no entry is ever merged into another.

    That is deliberate, and it means the eight entries are not eight distinct
    matrices. ``rmd:primary`` is bit-identical to
    ``class_conditional_full:primary``, and ``rmd:background`` to
    ``marginal_full:primary`` -- RMD *is* the difference of the two cells in the
    Full column, which is the decomposition the thesis reports, so the same
    matrix is fitted twice under two names. The fits are kept independent rather
    than shared: ``fit_scorer`` stays self-contained, and coupling two scorers
    through a shared mutable fit would be a worse trade than ~8 s of a ~33 s
    fit. ``summary["n_distinct_covariances"]`` reports how many underlying
    matrices the eight entries amount to, so a reader who sees the same
    condition number twice can tell that from a coincidence.

    **Scorers with no fitted covariance are skipped, not rejected.** kNN has no
    parameters to estimate and the PCA-residual scorer fits a subspace
    rather than a covariance, so a mixed dict of every scorer in the run is a
    perfectly reasonable thing to hand this function, and it is what a full run
    will have. Filtering here rather than raising is what lets
    ``run_manifest`` take one collection and split it itself.
    """
    # Test the *type*, not the attribute name. `primary` is a plausible name for
    # a PCA-residual scorer's fitted subspace (that family has two subspaces, a
    # natural primary/secondary pair, and GaussianScorer sets the precedent), and
    # such a scorer would pass a `hasattr(s, "primary")` filter and then die on
    # `scorer.background` -- the same AttributeError this filter exists to prevent.
    scorer_list = [
        s for s in _as_list(scorers)
        if isinstance(getattr(s, "primary", None), CovarianceFit)
    ]

    covariances: dict[str, Any] = {}
    per_scorer: dict[str, Any] = {}
    #: Content digests of (mean, covariance), so entries that are the same
    #: matrix under two names can be counted once without merging them.
    digests: set[tuple[Any, ...]] = set()
    for scorer in scorer_list:
        fits = [scorer.primary] + ([scorer.background] if scorer.background else [])
        for fit in fits:
            if fit.name in covariances:
                raise ValueError(
                    f"two covariance fits share the name {fit.name!r}; the second "
                    f"would silently overwrite the first and the manifest would "
                    f"under-report. Mirrors the duplicate-scorer-name guard in "
                    f"run_manifest. Unreachable through fit_scorer, which derives "
                    f"names from config names; reachable when fits are built by hand."
                )
            covariances[fit.name] = covariance_manifest_entry(fit)
            digests.add(_fit_digest(fit))
        per_scorer[scorer.name] = {
            "hyperparams": scorer.config.encoded_hyperparams(shrinkage=scorer.shrinkage),
            "covariances": [f.name for f in fits],
        }

    intensities = {
        k: v["shrinkage_intensity"] for k, v in covariances.items() if v["shrinkage"]
    }
    conditions = {k: v["condition_number"] for k, v in covariances.items()}

    return {
        "covariances": covariances,
        "scorers": per_scorer,
        "summary": {
            "n_covariances": len(covariances),
            "n_distinct_covariances": len(digests),
            "all_symmetric": all(v["is_symmetric"] for v in covariances.values()),
            "condition_numbers": conditions,
            "condition_number_max": max(conditions.values()) if conditions else None,
            "shrinkage_intensities": intensities,
        },
    }


def run_manifest(
    scorers: Mapping[str, Any] | Iterable[Any],
    *,
    seed: int,
    repo_commit: str,
    openood_commit: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The whole run's manifest: every scorer, every family, one fragment.

    This is what a full run writes. ``covariance_manifest`` covers only the
    Gaussian family's fitted matrices; a run also has kNN (no covariance, but a
    reference-set size and a k that both need recording) and
    the PCA-residual scorer (whose retained dimension *d* is one of the three
    quantities that must be read at runtime and never typed into a source file).

    Every scorer contributes its own ``diagnostics()``, so a new family is
    included by implementing that method rather than by editing this function.

    ``seed``, ``repo_commit`` and ``openood_commit`` are **required keyword
    arguments**, not validated keys. Two commit hashes are required in every
    manifest, and the bootstrap records its seed there because the seed plus the
    procedure is what reproduces the index sets; making them parameters means a
    caller that omits one gets a ``TypeError`` at the call site rather than a
    manifest that looks complete and silently is not. Nothing here invents them
    -- a manifest recording a guessed hash is worse than one recording none, but
    nothing here lets a run reach disk without them either. A scoring run is the
    only stage that cannot re-derive these afterwards, and losing them costs a
    refit of every covariance in the study.

    ``extra`` is merged at the top level and is for everything else worth
    recording: the embedding-cache manifest SHA, library versions, thread count.
    It is also where a scorer's **declared requirements** are satisfied: any key
    a scorer names in ``MANIFEST_REQUIRES`` must be present in ``extra`` or this
    raises. That covers quantities ``diagnostics()`` cannot produce because they
    depend on the data the scorer was run against -- the PCA-residual family's
    degeneracy rho being the one that forced it, since ``degeneracy_report``
    needs the scored
    embeddings and this function never sees them. See
    ``_assert_declared_requirements``.
    """
    scorer_list = _as_list(scorers)
    names = [s.name for s in scorer_list]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            f"duplicate scorer name(s) {duplicates}; manifest entries would overwrite "
            f"each other, and so would the score-dump columns."
        )

    _assert_declared_requirements(scorer_list, extra)

    fragment: dict[str, Any] = {
        "scorers": {s.name: s.diagnostics() for s in scorer_list},
        "n_scorers": len(scorer_list),
        "seed": seed,
        "repo_commit": repo_commit,
        "openood_commit": openood_commit,
    }
    covariance_part = covariance_manifest(scorer_list)
    if covariance_part["summary"]["n_covariances"]:
        fragment["covariances"] = covariance_part["covariances"]
        fragment["covariance_summary"] = covariance_part["summary"]
    if extra:
        fragment.update(dict(extra))
    return fragment


def _assert_declared_requirements(
    scorer_list: list[Any], extra: Mapping[str, Any] | None
) -> None:
    """Enforce every key the scorers declare through ``MANIFEST_REQUIRES``.

    **This function knows nothing about any particular family**, which is the
    point. Some quantities cannot be collected by ``diagnostics()`` because they
    are functions of a scorer *and* of the data it was run on, and
    ``run_manifest`` receives scorers only -- the PCA-residual family's
    degeneracy rho is the case that forced this. Rather than teaching the manifest about
    PCA-residual scorers, a scorer declares what its manifest must carry and
    this enforces the declaration, exactly as a family joins the manifest by
    implementing ``diagnostics()`` rather than by editing this module. A later
    family with its own unrecoverable quantity declares it and is guarded for
    free.

    Presence only. Nothing here inspects or invents a value: a manifest
    recording a guessed rho would be worse than one recording none, and these
    runtime-measured quantities must stay runtime values.
    """
    supplied = set(extra or {})
    missing: dict[str, list[str]] = {}
    for scorer in scorer_list:
        for key in getattr(scorer, "MANIFEST_REQUIRES", ()):
            if key not in supplied:
                missing.setdefault(key, []).append(scorer.name)
    if missing:
        detail = "; ".join(
            f"{key!r} (required by {', '.join(sorted(names))})"
            for key, names in sorted(missing.items())
        )
        raise ValueError(
            f"run_manifest is missing declared key(s) in extra=: {detail}. "
            f"These are quantities a scorer's diagnostics() cannot produce on "
            f"its own because they depend on the data the scorer was run "
            f"against, so they have to be computed by the caller and passed "
            f"in. For the PCA-residual family that is "
            f"degeneracy_report(scorer, embeddings) "
            f"-> extra={{'degeneracy': ...}}, computed on the embeddings being "
            f"scored. Omitting it writes a manifest that looks complete and "
            f"leaves the pre-declared threshold with no run to apply to."
        )


def write_run_manifest(
    path: str | Path,
    scorers: Mapping[str, Any] | Iterable[Any],
    *,
    seed: int,
    repo_commit: str,
    openood_commit: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``run_manifest`` to ``path`` as JSON and return it.

    The three provenance arguments are required here for the same reason they
    are required there: this is the function that puts a manifest on disk.
    """
    fragment = run_manifest(
        scorers,
        seed=seed,
        repo_commit=repo_commit,
        openood_commit=openood_commit,
        extra=extra,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fragment, indent=2, sort_keys=True) + "\n")
    return fragment

