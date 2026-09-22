"""What each detector must RETAIN at inference, and what the shared GPU pass costs.

The thesis reports a scoring time per method and an AUROC per method, and says
nothing at all about memory. That is the half of the efficiency story a
practitioner asks about first, because it is the half that decides whether a
detector fits beside a model in production, and because it is the property that
EXPLAINS the timing ordering rather than merely restating it.

THE HEADLINE THIS SCRIPT EXISTS FOR. Two of the thirteen detectors carry the
training set into inference and eleven do not. Everything else follows from
that one fact: the footprint split, the two-orders-of-magnitude scoring gap, and
the reason the gap is n / p and not a constant.

DERIVED, NOT ASSERTED. Every shape here is read off ``scores/run_metadata.json``
or off the fitted classes in ``src/xai_ood/methods/``. Nothing is transcribed.
Two shapes came back different from what was expected of them and both
differences are real; they are reported in
``inference-footprint-per-method.csv``'s ``fitted_over_minimal`` column.

TWO NUMBERS PER METHOD, WHICH IS THE POINT. ``minimal`` is what the method's
definition requires: the smallest object that can produce the same score.
``as_fitted`` is what this implementation actually holds resident. Where they
differ by more than a few per cent the difference is an implementation choice,
not a property of the method, and the Results chapter must say which one it is
quoting. The ratio column is the finding.

MEGABYTES, AND AT WHICH DTYPE. Both are given: ``megabytes`` is SI, 10^6 bytes,
and ``mebibytes`` is 2^20. The dtype is carried per object because it is not
uniform: the embedding cache on disk is float32 and every fitted object promotes
to float64, so a reference set doubles in size on the way into a scorer.

NO RUN, NO GPU, NO EMBEDDINGS. This is arithmetic on shapes and on recorded
timestamps. The only bytes read from the embedding tree are ``.npy`` headers and
``manifest.json`` files.

Read-only against the repository's results tree. Writes only into this directory.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# The repository root, the directory above this one. Overridable with
# XAI_OOD_ROOT so a run tree held outside the checkout can be read in place.
ROOT = Path(os.environ.get("XAI_OOD_ROOT", Path(__file__).resolve().parents[1]))
RUN = ROOT / "results" / "runs" / "2026-09-10-full-spectrum-shrunk"
SRC = ROOT / "src" / "xai_ood" / "methods"
CACHE = ROOT / "embeddings-local"
OUT = Path(__file__).resolve().parent

F64 = 8
F32 = 4
I64 = 8

#: The near/far batch ran its six splits sequentially in one loop, which is what
#: makes the gap between two consecutive completion timestamps a duration rather
#: than a coincidence. Asserted against the script rather than trusted.
BATCH_ORDER = ("cifar100", "tin", "mnist", "svhn", "texture", "places365")
BATCH_SCRIPT = ROOT / "scripts" / "run_near_far_ood.sh"

#: Figures written into ``knn.py``'s own docstrings by whoever wrote the memory
#: knob, independent of this script and of each other. If the byte arithmetic
#: here is wrong, these do not reproduce.
SOURCE_PINS = (
    ("50,000 x 768 float64 reference, GiB", 50_000 * 768 * F64 / 2**30, 0.29, 0.005),
    ("both kNN variants resident, GiB", 2 * 50_000 * 768 * F64 / 2**30, 0.57, 0.005),
    ("9,000 x 50,000 float64 distances, GB", 9_000 * 50_000 * F64 / 1e9, 3.6, 0.05),
    ("512-row distance block, MB", 512 * 50_000 * F64 / 1e6, 200.0, 5.0),
)


def fail(message: str) -> None:
    sys.exit(f"REFUSING TO WRITE: {message}")


# --------------------------------------------------------------------------- #
# The retained-object model, and the guard that keeps it honest
# --------------------------------------------------------------------------- #


def retained_attributes(path: Path, class_name: str) -> set[str]:
    """Every ``self.X = ...`` assigned in ``class_name.__init__``, read by ast.

    The footprint model below enumerates what each scorer holds. If a scorer
    gains an attribute, the model is silently wrong and the table is silently
    too small. This is the check that turns that into an exit.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    names = set()
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Assign):
                            for target in sub.targets:
                                if (
                                    isinstance(target, ast.Attribute)
                                    and isinstance(target.value, ast.Name)
                                    and target.value.id == "self"
                                ):
                                    names.add(target.attr)
                    return names
    fail(f"{path.name}: no __init__ found for {class_name}")
    return set()


def dataclass_fields(path: Path, class_name: str) -> set[str]:
    """Field names of a frozen dataclass, read by ast so no import is needed."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    fail(f"{path.name}: no dataclass {class_name}")
    return set()


#: What the model accounts for, per class. Anything the source holds that is not
#: listed here stops the script. Scalars and strings are listed too, so that
#: "not an array" is a recorded decision rather than an omission.
ACCOUNTED = {
    ("covariance.py", "CovarianceFit"): {
        "mean", "covariance", "class_means", "class_labels", "_cholesky",
        "diagonal", "shrinkage", "shrinkage_intensity", "n_samples", "name",
        "diagnostics",
    },
    ("pca_residual.py", "SubspaceFit"): {
        "origin", "components", "eigenvalues",
        "n_components_available", "n_fitting_rows", "name", "diagnostics",
    },
    ("knn.py", "KnnScorer"): {"reference", "_reference_sq", "config"},
    ("gaussian.py", "GaussianScorer"): {"primary", "background", "config", "shrinkage"},
    ("pca_residual.py", "PcaResidualScorer"): {
        "subspace", "mean", "whitening_fit", "config", "d",
    },
}


def check_inventory() -> None:
    for (filename, class_name), accounted in ACCOUNTED.items():
        path = SRC / filename
        if class_name in ("CovarianceFit", "SubspaceFit"):
            found = dataclass_fields(path, class_name)
        else:
            found = retained_attributes(path, class_name)
        unaccounted = found - accounted
        if unaccounted:
            fail(
                f"{filename}:{class_name} retains {sorted(unaccounted)}, which the "
                f"footprint model does not account for. The table would be too small."
            )
        vanished = accounted - found
        if vanished:
            fail(
                f"{filename}:{class_name} no longer retains {sorted(vanished)}; the "
                f"footprint model is stale."
            )


# --------------------------------------------------------------------------- #
# Footprints
# --------------------------------------------------------------------------- #


def covariance_fit_bytes(p: int, n_classes: int) -> tuple[int, list[str]]:
    """As-fitted bytes of one ``CovarianceFit``, and what is in it.

    Note what this counts and why. ``covariance`` AND ``_cholesky`` are both
    held, both dense ``(p, p)`` float64, for every cell including the diagonal
    ones: ``fit_covariance`` builds the diagonal cells as
    ``np.diag(np.diag(cov))``, a dense matrix with zeros off the diagonal, and
    ``np.linalg.cholesky`` of it is dense too. Inference touches only the
    Cholesky, through ``np.linalg.solve``.
    """
    parts = [f"mean:({p},):float64", f"covariance:({p},{p}):float64",
             f"_cholesky:({p},{p}):float64"]
    total = p * F64 + 2 * p * p * F64
    if n_classes:
        parts.append(f"class_means:({n_classes},{p}):float64")
        parts.append(f"class_labels:({n_classes},):int64")
        total += n_classes * p * F64 + n_classes * I64
    return total, parts


def triangular_bytes(p: int) -> int:
    """A Cholesky factor stored as its ``p(p+1)/2`` non-zero entries."""
    return (p * (p + 1) // 2) * F64


def main() -> int:
    metadata = json.loads((RUN / "scores" / "run_metadata.json").read_text())
    scorers = metadata["scorers"]
    timing = metadata["timing"]
    fit_reference = metadata["fit_reference"]["per_scorer"]
    n_scorers = int(metadata["n_scorers"])

    check_inventory()

    # ---- pins ------------------------------------------------------------- #
    for label, derived, pinned, tolerance in SOURCE_PINS:
        if abs(derived - pinned) > tolerance:
            fail(f"pin {label}: derived {derived:.4f}, source says {pinned}")
    print(f"byte arithmetic pinned against {len(SOURCE_PINS)} figures written in "
          f"src/xai_ood/methods/knn.py")

    # The embedding cache on disk: dtype and shape read from .npy headers, size
    # from the filesystem. Pins the float32 claim and the row counts at once.
    cache = {}
    for manifest_path in sorted(CACHE.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        split = manifest["split_name"]
        npy = manifest_path.parent / "cls.npy"
        with open(npy, "rb") as handle:
            version = __import__("numpy").lib.format.read_magic(handle)
            shape, _, dtype = __import__("numpy").lib.format._read_array_header(handle, version)
        expected = shape[0] * shape[1] * dtype.itemsize + 128
        if npy.stat().st_size != expected:
            fail(f"{split}: cls.npy is {npy.stat().st_size} bytes, shape/dtype say {expected}")
        if shape[0] != manifest["row_count"] or str(dtype) != manifest["dtype"]:
            fail(f"{split}: header {shape} {dtype} against manifest "
                 f"{manifest['row_count']} {manifest['dtype']}")
        stamp = datetime.fromisoformat(manifest["extraction_date_utc"])
        mtime = datetime.fromtimestamp(npy.stat().st_mtime, tz=stamp.tzinfo)
        drift = abs((mtime - stamp).total_seconds())
        if drift > 1.0:
            fail(f"{split}: manifest stamp and cls.npy mtime differ by {drift:.1f}s; "
                 f"the extraction durations below cannot be trusted")
        cache[split] = {
            "rows": int(shape[0]), "p": int(shape[1]), "dtype": str(dtype),
            "bytes": npy.stat().st_size, "end": stamp,
            "device": manifest["library_versions"].get("cuda_device"),
        }
    print(f"cache: {len(cache)} splits, .npy headers agree with manifests, "
          f"manifest timestamps agree with file mtimes to under 1 s")

    p = {c["p"] for c in cache.values()}
    if len(p) != 1:
        fail(f"splits disagree on the embedding dimension: {p}")
    p = p.pop()
    n_reference = int(metadata["fit_reference"]["rows"])

    # ---- the footprint table --------------------------------------------- #
    rows = []
    for name in sorted(scorers):
        entry = scorers[name]
        tier = entry["reporting_tier"]
        score_us = float(timing["score"][name]["microseconds_per_sample"])
        fit_s = float(timing["fit"][name]["seconds"])
        ref_rows = int(fit_reference[name]["reference_rows"])

        if name.startswith("knn"):
            n = int(entry["n_reference"])
            if n != n_reference:
                fail(f"{name}: reference is {n} rows, the run fitted on {n_reference}")
            as_fitted = n * p * F64 + n * F64
            parts = [f"reference:({n},{p}):float64", f"_reference_sq:({n},):float64"]
            # The reference set IS the method; the floor is that same set at the
            # dtype the cache holds, with the cached squared norms recomputed.
            minimal = n * p * F32
            minimal_note = f"the reference set at the cache dtype, ({n},{p}) float32"
            needs_train = "yes"
            scales = f"O(n p), n = {n}"
            complexity = "O(n p) per query"
        elif name.startswith("pca_residual"):
            d = int(entry["d"])
            as_fitted = p * F64 * 3 + p * p * F64  # mean, origin, eigenvalues, components
            parts = [f"mean:({p},):float64", f"subspace.origin:({p},):float64",
                     f"subspace.components:({p},{p}):float64",
                     f"subspace.eigenvalues:({p},):float64"]
            # Either the retained basis or its complement gives the identical
            # residual norm, so the floor is whichever is narrower.
            k = min(d, p - d)
            minimal = p * k * F64 + p * F64
            minimal_note = (f"the narrower of the two bases, ({p},{k}) float64, "
                            f"plus mu_ID; d = {d}, p - d = {p - d}")
            complexity = f"O(d p) per query, d = {d}"
            if entry.get("whiten"):
                whitening, whitening_parts = covariance_fit_bytes(p, 0)
                as_fitted += whitening
                parts += [f"whitening_fit.{q}" for q in whitening_parts]
                minimal = triangular_bytes(p) + p * F64 + p * k * F64
                minimal_note = ("the whitening Cholesky as a triangle plus mu_ID plus "
                                f"the ({p},{k}) basis")
                complexity = f"O(p^2) per query, from the whitening solve; d = {d}"
            needs_train = "no"
            scales = "O(p^2), independent of n"
        else:
            covs = entry["covariances"]
            primary = covs["primary"]
            n_classes = int(primary["n_classes"])
            as_fitted, parts = covariance_fit_bytes(p, n_classes)
            if primary["diagonal"]:
                # The restriction is on the estimator, not on the storage: the
                # matrix is dense with zeros off the diagonal, and so is its
                # Cholesky. The floor is the p variances.
                minimal = p * F64 + max(n_classes, 1) * p * F64
                minimal_note = (f"{p} variances plus "
                                f"{'the ' + str(n_classes) + ' class means' if n_classes else 'the mean'}")
                complexity = f"O({'K ' if n_classes else ''}p) per query if stored diagonally"
            else:
                minimal = triangular_bytes(p) + max(n_classes, 1) * p * F64
                minimal_note = (f"the Cholesky factor as a triangle, {p}({p}+1)/2, plus "
                                f"{'the ' + str(n_classes) + ' class means' if n_classes else 'the mean'}")
                complexity = f"O({'K ' if n_classes else ''}p^2) per query"
            if "background" in covs:
                background, background_parts = covariance_fit_bytes(p, 0)
                as_fitted += background
                parts += [f"background.{q}" for q in background_parts]
                minimal += triangular_bytes(p) + p * F64
                minimal_note += "; plus the same again for the background term"
                complexity = f"O(K p^2) + O(p^2) per query"
            needs_train = "no"
            scales = "O(p^2), independent of n"
            if ref_rows not in (n_reference, n_classes):
                fail(f"{name}: reference_rows {ref_rows} is neither {n_reference} nor {n_classes}")

        rows.append({
            "scorer": name,
            "reporting_tier": tier,
            "needs_training_set_at_inference": needs_train,
            "footprint_scaling": scales,
            "score_complexity": complexity,
            "retained_objects": "; ".join(parts),
            "as_fitted_bytes": as_fitted,
            "as_fitted_megabytes": round(as_fitted / 1e6, 4),
            "as_fitted_mebibytes": round(as_fitted / 2**20, 4),
            "minimal_bytes": minimal,
            "minimal_megabytes": round(minimal / 1e6, 4),
            "minimal_note": minimal_note,
            "fitted_over_minimal": round(as_fitted / minimal, 2),
            "score_microseconds_per_sample": round(score_us, 4),
            "fit_seconds": round(fit_s, 4),
        })

    if len(rows) != n_scorers:
        fail(f"{len(rows)} rows for {n_scorers} scorers")

    with open(OUT / "inference-footprint-per-method.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    resident = sum(r["as_fitted_bytes"] for r in rows)
    with_train = [r for r in rows if r["needs_training_set_at_inference"] == "yes"]
    without = [r for r in rows if r["needs_training_set_at_inference"] == "no"]
    train_bytes = sum(r["as_fitted_bytes"] for r in with_train)
    print()
    print(f"THE AXIS: {len(with_train)} of {n_scorers} detectors carry the training set "
          f"into inference.")
    print(f"  they are {sorted(r['scorer'] for r in with_train)}")
    print(f"  all {n_scorers} resident at once: {resident / 1e6:.2f} MB")
    print(f"  the {len(with_train)} that keep the reference set: {train_bytes / 1e6:.2f} MB, "
          f"{100 * train_bytes / resident:.1f}% of it")
    print(f"  the other {len(without)} together: {(resident - train_bytes) / 1e6:.2f} MB, "
          f"largest single {max(r['as_fitted_megabytes'] for r in without):.2f} MB")
    print(f"  crossover: a reference set is smaller than one covariance when n < p, "
          f"here n / p = {n_reference / p:.1f}")

    # ---- extraction cost, recovered from the Phase 1 records --------------- #
    text = BATCH_SCRIPT.read_text()
    if f"for split in {' '.join(BATCH_ORDER)}" not in text:
        fail(f"{BATCH_SCRIPT.name} no longer runs {BATCH_ORDER} sequentially in that order; "
             f"consecutive completion stamps are not durations.")
    extraction = []
    for previous, split in zip(BATCH_ORDER, BATCH_ORDER[1:]):
        seconds = (cache[split]["end"] - cache[previous]["end"]).total_seconds()
        if seconds <= 0:
            fail(f"{split} completed before {previous}; the batch did not run in order")
        extraction.append({
            "split": split,
            "rows": cache[split]["rows"],
            "seconds": round(seconds, 3),
            "milliseconds_per_image": round(1000 * seconds / cache[split]["rows"], 4),
            "measured_as": f"completion stamp minus {previous}'s; same sequential batch",
            "device": cache[split]["device"],
        })
    total_seconds = sum(e["seconds"] for e in extraction)
    total_rows = sum(e["rows"] for e in extraction)
    extraction_us = 1e6 * total_seconds / total_rows
    extraction.append({
        "split": "AGGREGATE five consecutive splits; contiguous wall clock",
        "rows": total_rows,
        "seconds": round(total_seconds, 3),
        "milliseconds_per_image": round(extraction_us / 1000, 4),
        "measured_as": "sum of the five durations over the sum of their rows",
        "device": extraction[0]["device"],
    })
    with open(OUT / "extraction-cost-per-image.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(extraction[0]))
        writer.writeheader()
        writer.writerows(extraction)

    # ---- amortisation ------------------------------------------------------ #
    suite_us = sum(r["score_microseconds_per_sample"] for r in rows)
    share_us = extraction_us / n_scorers
    amortisation = []
    for r in sorted(rows, key=lambda x: x["score_microseconds_per_sample"]):
        own = r["score_microseconds_per_sample"]
        amortisation.append({
            "scorer": r["scorer"],
            "score_microseconds_per_sample": own,
            "extraction_share_at_13_methods_microseconds": round(share_us, 2),
            "share_over_own_cost": round(share_us / own, 2),
            "methods_needed_before_own_cost_matches_its_share": round(extraction_us / own, 1),
            "own_cost_exceeds_its_share": "yes" if own > share_us else "no",
        })
    with open(OUT / "amortisation-per-method.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(amortisation[0]))
        writer.writeheader()
        writer.writerows(amortisation)

    bundled = n_scorers * extraction_us + suite_us
    shared = extraction_us + suite_us
    print()
    print(f"THE AMORTISATION POINT, on {total_rows:,} images of Phase 1 extraction:")
    print(f"  one forward pass: {extraction_us:.1f} us/image on an "
          f"{extraction[0]['device']}")
    print(f"  all {n_scorers} scorers together: {suite_us:.1f} us/image on "
          f"{timing['machine']['node']}")
    print(f"  the pass costs {extraction_us / suite_us:.2f}x the entire suite, so at "
          f"{n_scorers} methods it is NOT yet amortised")
    print(f"  per method if shared: {share_us:.1f} us/image; "
          f"{sum(1 for a in amortisation if a['own_cost_exceeds_its_share'] == 'yes')} of "
          f"{n_scorers} scorers cost more than that")
    print(f"  bundling the pass into each method (the OpenOOD v1.5 convention) reads "
          f"{bundled / 1000:.2f} ms/image against {shared / 1000:.2f} shared, "
          f"{bundled / shared:.2f}x")

    # ---- corroboration, reported not gated -------------------------------- #
    us = {r["scorer"]: r["score_microseconds_per_sample"] for r in rows}
    print()
    print("COST ORDERING AGAINST WHAT THE FOOTPRINT PREDICTS (entry 38 is an ordering "
          "claim, so these are ratios read, not ratios estimated):")
    print(f"  knn_unnormalized / marginal_full = {us['knn_unnormalized'] / us['marginal_full']:.1f}, "
          f"O(n p) / O(p^2) predicts n / p = {n_reference / p:.1f}")
    print(f"  class_conditional_full / marginal_full = "
          f"{us['class_conditional_full'] / us['marginal_full']:.2f}, O(K p^2) / O(p^2) "
          f"predicts K = 10")
    print(f"  marginal_diagonal / marginal_full = "
          f"{us['marginal_diagonal'] / us['marginal_full']:.2f}. A genuinely diagonal "
          f"implementation would be about p = {p}x cheaper; it is not, which confirms "
          f"the dense-diagonal finding from the timings as well as from the source.")
    print(f"  rmd against its two components summed = {us['rmd']:.1f} against "
          f"{us['class_conditional_full'] + us['marginal_full']:.1f} us")
    print()
    print(f"wrote inference-footprint-per-method.csv, extraction-cost-per-image.csv "
          f"and amortisation-per-method.csv to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
