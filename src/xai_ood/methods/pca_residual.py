"""PCA projection-residual scoring, two subspaces, reported separately.

OpenOOD's ``Residual``/``ViM`` postprocessors are not usable here: both call
``net.get_fc()`` for a classifier weight and bias and define the subspace origin
as ``u = -pinv(w) @ b``, and ViM's score is roughly half logit energy. Neither
has a head-free form usable on frozen embeddings. This module is head-free by
construction.

The score, identical for both subspaces
---------------------------------------
With ``P_d`` the orthogonal projector onto the retained ``d``-dimensional
principal subspace::

    r_d(z) = || (I - P_d) (z - mu_ID) ||_2

Higher = more OOD, in line with the package rule; nothing is negated anywhere.
**No per-subspace rescaling**, so the two components are directly comparable --
and they are reported **separately and never combined into one number.**

The two subspaces
-----------------
* ``all_id`` -- PCA on all 50,000 ID *train* embeddings. Coincides with a
  probe-free reduction of OpenOOD's ``Residual``, replacing ``-pinv(w) b`` with
  the ID training mean as the subspace origin.
* ``class_mean`` -- PCA on the K per-class mean embeddings. The component that
  most directly tests GradPCA's intuition on raw embeddings.

Both are fit on **train, not val**, centred on ``mu_ID``, for consistency with
MDS, RMDS and kNN. OpenOOD's ``residual_postprocessor.py`` fits on ``val``; that
deviation is kept only where published numbers are being reproduced, and the
switch is documented rather than silent. Eigendecomposition is ``eigh``, never
``eig`` -- enforced, not merely intended, by
``tests/test_import_hygiene.py::test_no_forbidden_numpy_linalg_routine_in_the_package``.

Retained dimension
------------------
* ``all_id``: **d is read off the eigenspectrum at runtime** by the
  95%-explained-variance rule. It is one of the three quantities that can only
  be measured once the embeddings exist, and it is **never a constant in this
  file**. The residual
  is over the remaining ``768 - d`` minor-variance directions.
  ``SWEEP_D`` is reported as an appendix sensitivity figure and is **never
  used to reselect d** for the headline table.
* ``class_mean``: ``d = K - 1`` (9 for CIFAR-10), every available component.
  Nothing to sweep -- the centred class-mean matrix has rank at most ``K - 1``
  for any class balance, because ``sum_k n_k (mu_k - mu_ID) = 0`` is a linear
  dependency among its rows by the definition of ``mu_ID``.

Variants
--------
Unnormalized is **primary**; L2-normalized is an **ablation**. MaRS reports
normalization hurting a residual-space score while Mahalanobis++ reports it
helping raw-embedding Mahalanobis, so the direction is an empirical question
here rather than an assumption. Normalization goes through
``xai_ood.methods.covariance.l2_normalize``, the codebase's one guarded
implementation; this module does not have its own.

Whitening before projection is the remedy if the degeneracy control fires, and
ships as a **separately labeled variant** -- never silently substituted for the
primary. See ``degeneracy_report``, where the threshold and its timing live.

Baselines, kept apart
---------------------
1. **Degeneracy control**, "is this a subspace method at all?":
   ``degeneracy_report``. Spearman rho of the class-mean residual against
   the plain norm and against the centred norm, the latter being the sharper
   control since the residual is computed on centred vectors. Threshold at
   ``DEGENERACY_RHO_THRESHOLD``.
2. **Comparative baseline**, "does it beat the alternative?": the
   ``marginal_diagonal`` cell of the 2x2 (``xai_ood.methods.gaussian``).
   This family's headline claim is the paired AUROC *difference* against that
   cell
   under the cluster bootstrap, not a bare AUROC in a table.

Citation framing: Wang et al. (2022) for the residual-norm idea, Guan et al.
(2023) as direct precedent for PCA-based OOD detection, GradPCA (Seleznova et
al., 2026) as the conceptual inspiration for this family, MaRS (Di
Salvo et al., 2026) as same-institution work on the same mechanism. Positioned
inside a well-populated family, not as novel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..reporting import validate_reporting_tier
from ..schema import encode_hyperparams
from .correlation import spearman_rho
from .covariance import CovarianceFit, fit_covariance, l2_normalize

__all__ = [
    "VARIANCE_THRESHOLD",
    "DEGENERACY_RHO_THRESHOLD",
    "SWEEP_D",
    "SUBSPACES",
    "SubspaceFit",
    "PcaResidualConfig",
    "PcaResidualScorer",
    "CONFIGURATIONS",
    "config_by_name",
    "explained_variance_ratio",
    "select_dimension_by_variance",
    "fit_pca_residual",
    "fit_all_pca_residual",
    "degeneracy_report",
]


#: The 95%-explained-variance rule for the all-ID subspace. A **fixed decision
#: rule**, which is what closes the held-out-set violation: d is not tuned
#: against the test AUROC, it is read off the ID-train eigenspectrum by a rule
#: that does not consult it. The rule is a constant; the integer it produces is
#: not, and must never be written into a source file. See
#: ``select_dimension_by_variance``.
VARIANCE_THRESHOLD: float = 0.95

#: The degeneracy threshold, also fixed in advance: if Spearman rho between the
#: class-mean residual and the **centred** norm reaches this, the class-mean
#: component is written up as a rescaled-norm score rather than as subspace
#: structure, and the whitened variant is the declared remedy. Same distinction
#: as above -- the threshold is a fixed decision and belongs in the source; the
#: measured rho is a runtime measurement and does not.
DEGENERACY_RHO_THRESHOLD: float = 0.95

#: Appendix sensitivity grid for the all-ID subspace. **Reported, never used to
#: reselect d.** Same pre-declared discipline as kNN's ``SWEEP_K``: sweeping
#: against the test AUROC being reported would hand this family a free parameter
#: the other three families do not have. ``768`` is the full-rank endpoint,
#: where ``P_d = I`` and the residual is identically zero -- a legitimate and
#: informative end of the curve, not a bug.
SWEEP_D: tuple[int, ...] = (16, 32, 64, 128, 256, 384, 512, 768)

#: The two subspaces, reported separately and **never combined**.
SUBSPACES: tuple[str, ...] = ("all_id", "class_mean")

#: Query rows per projection block. Mirrors ``knn._QUERY_BLOCK``: purely a
#: memory knob, results identical for any block size. The sweep materialises an
#: ``(n, p)`` coefficient matrix, 55 MB at 9,000 x 768, so this matters much
#: less here than it does for kNN's 9,000 x 50,000 distance matrix.
_QUERY_BLOCK: int = 4096


# --------------------------------------------------------------------------- #
# The variance rule
# --------------------------------------------------------------------------- #


def explained_variance_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """Cumulative fraction of total variance explained by the first d components.

    ``eigenvalues`` are expected **descending**, as ``SubspaceFit`` stores
    them. Element ``i`` is the fraction explained by components ``1..i+1``, so
    ``d`` components correspond to index ``d - 1``.

    Divisor-invariant: scaling every eigenvalue by a constant leaves the ratios
    unchanged, which is why it does not matter that
    ``xai_ood.methods.covariance.fit_covariance`` divides by ``n`` (the MLE)
    rather than ``n - 1``.
    """
    values = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("explained_variance_ratio: no eigenvalues")
    if (values < 0).any():
        # Tiny negatives from round-off are expected on a rank-deficient matrix
        # and are clipped by _pca_basis before this is ever called. Reaching
        # here with a genuinely negative eigenvalue means the input is not a
        # covariance spectrum.
        raise ValueError(
            "explained_variance_ratio: negative eigenvalue(s); a covariance "
            "spectrum has none. Round-off negatives are clipped at fit time, so "
            "this is a real one."
        )
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError(
            "explained_variance_ratio: total variance is zero, so no fraction of "
            "it is defined. Every fitted direction is degenerate."
        )
    return np.cumsum(values) / total


def select_dimension_by_variance(
    eigenvalues: np.ndarray, *, threshold: float = VARIANCE_THRESHOLD
) -> int:
    """Smallest d whose first d components explain ``threshold`` of the variance.

    **This is the function that produces the integer d.** It is evaluated
    against the real ID-train eigenspectrum only at run time and its answer is
    unknown until then; nothing here or anywhere else in ``src/`` may hardcode a
    guess. ``tests/test_pca_residual.py`` checks it against constructed spectra
    whose answers are arithmetic, never against a remembered value.

    The threshold comparison is ``>=``, so a spectrum explaining exactly 95% at
    d selects that d rather than d + 1. Stated because it is the kind of
    boundary that is decided once by accident and then argued about later;
    "components explaining 95% of the variance" reads inclusively.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            f"select_dimension_by_variance: threshold must be in (0, 1], got {threshold}"
        )
    ratio = explained_variance_ratio(eigenvalues)
    reached = np.flatnonzero(ratio >= threshold)
    if reached.size == 0:
        # Unreachable for threshold <= 1 in exact arithmetic, since the last
        # entry is 1.0 by construction; possible at exactly 1.0 through
        # round-off. Fall back to every component rather than raise: "all of
        # them" is the right answer to "how many explain everything".
        return int(ratio.size)
    return int(reached[0]) + 1


# --------------------------------------------------------------------------- #
# The subspace fit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SubspaceFit:
    """A fitted principal subspace: an orthonormal basis and its spectrum.

    ``components`` is ``(p, p)`` with **eigenvectors in columns, descending by
    eigenvalue** -- a complete basis, not just the retained ``d`` of it, because
    ``PcaResidualScorer.score_sweep`` needs the whole spectrum of
    coefficients to produce every d in the grid from one pass.

    ``n_components_available`` is the **numerical rank**, which for the
    class-mean subspace is at most ``K - 1`` and is the quantity that would
    reveal an unbalanced or collapsed class structure. It is recorded rather
    than asserted, since it is a property of the data, not of the code.

    ``origin`` is the point the fitting data was centred on, expressed in the
    **projection space**: ``mu_ID`` for the unwhitened variants, and exactly
    zero for the whitened ones, where the whitening has already subtracted it.
    It is recorded for the manifest and for inspection;
    ``PcaResidualScorer._prepare`` is the one place centring happens, so the
    two paths cannot disagree about where the origin is.
    """

    origin: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    n_components_available: int
    n_fitting_rows: int
    name: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.components.shape[0])


def _pca_basis(rows: np.ndarray, mean: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Eigendecomposition of the second-moment matrix of ``rows - mean``.

    Returns ``(eigenvalues_descending, eigenvectors_descending, numerical_rank)``.

    One route for both subspaces, deliberately. The all-ID case could equally
    take the eigenvectors of the marginal ``CovarianceFit`` and the class-mean
    case could take an SVD of a 10 x 768 matrix; using ``eigh`` on the ``(p, p)``
    second-moment matrix in both means the subspaces differ only in *what is fed
    in*, exactly as the score definition differs only in ``d``. At p = 768 the
    decomposition costs a fraction of a second and runs once per fit.

    ``eigh`` returns ascending eigenvalues; they are reversed here so that
    "component 1" is the leading one everywhere in this module. Round-off
    negatives on the rank-deficient class-mean matrix are clipped to zero -- the
    matrix is positive semi-definite by construction, so a negative eigenvalue
    is numerical noise and never information.
    """
    centred = np.asarray(rows, dtype=np.float64) - mean
    n_rows = centred.shape[0]
    # Divisor n, matching fit_covariance's MLE convention. Immaterial to the
    # eigenvectors and to the variance ratios, both of which are scale-free;
    # kept consistent so the eigenvalues reported in the manifest are on the
    # same footing as the covariance diagnostics beside them.
    second_moment = centred.T @ centred / n_rows
    second_moment = 0.5 * (second_moment + second_moment.T)

    eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    largest = float(eigenvalues[0])
    if largest <= 0.0:
        raise ValueError(
            f"{name}: the fitting data has zero variance in every direction, so "
            f"there is no principal subspace to project onto."
        )
    # Numerical rank at a relative tolerance of 1e-8 against the largest
    # eigenvalue: below that, a direction carries no variance worth retaining.
    rank = int((eigenvalues > 1e-8 * largest).sum())
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvalues, np.ascontiguousarray(eigenvectors), rank


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PcaResidualConfig:
    """One reported variant. Three factors and a name; no logic.

    ``subspace`` picks the fitting data, ``normalize`` the ablation,
    ``whiten`` the degeneracy remedy.

    ``reporting_tier`` is keyword-only with no default. See
    ``xai_ood.reporting``; the whitened variant is the one member of the
    ``conditional`` tier, meaning it runs unconditionally and is narrated only
    if the degeneracy control fires.
    """

    name: str
    subspace: str
    normalize: bool = False
    whiten: bool = False
    reporting_tier: str = field(kw_only=True)

    def __post_init__(self) -> None:
        validate_reporting_tier(self.reporting_tier, name=self.name)
        if self.subspace not in SUBSPACES:
            raise ValueError(
                f"{self.name}: unknown subspace {self.subspace!r}; the two "
                f"are {list(SUBSPACES)}. They are reported "
                f"separately and never combined."
            )
        if self.subspace == "all_id" and self.whiten:
            raise ValueError(
                f"{self.name}: the all-ID subspace cannot be whitened. Whitening "
                f"maps the ID cloud's covariance to exactly the identity, so its "
                f"principal subspace is degenerate -- every eigenvalue is 1 and "
                f"eigh returns a basis fixed by round-off, not by the data. "
                f"Measured at p = 40: spectrum std 3e-15, and a row-permuted "
                f"refit correlated 0.17 with itself. Whitening is pre-declared "
                f"only as the class-mean remedy, and for the all-ID subspace it "
                f"is not merely unreported but undefined."
            )

    @property
    def class_conditional(self) -> bool:
        """The class-mean subspace needs labels; the all-ID subspace does not."""
        return self.subspace == "class_mean"

    def hyperparams(self, *, d: int | None = None) -> dict[str, Any]:
        """The dict for the results table's ``hyperparams`` column.

        ``d`` is passed in from the *fitted* scorer rather than stored on the
        config, because for the all-ID subspace it is not known until the
        eigenspectrum has been computed. A config that carried a d would be a
        config that could carry a guessed one.
        """
        return {
            "subspace": self.subspace,
            "d": d,
            "d_rule": "explained_variance_95" if self.subspace == "all_id" else "k_minus_1",
            "normalize": self.normalize,
            "whiten": self.whiten,
            "centre": "mu_id",
            "fitting_split": "cifar10_train",
            "eigendecomposition": "eigh",
        }

    def encoded_hyperparams(self, *, d: int | None = None) -> str:
        return encode_hyperparams(self.hyperparams(d=d))


#: The reported variants. Names match the keys added to
#: ``xai_ood.visualization.style._SERIES_COLOR``.
#:
#: Five, not six -- and the sixth is **refused**, not merely unreported.
#: ``PcaResidualConfig`` raises on ``all_id`` + ``whiten``. Whitening makes
#: the ID cloud's covariance exactly the identity, so the "principal" subspace
#: of that cloud is degenerate and ``eigh`` returns a round-off-determined
#: basis; a row permutation, a no-op for PCA, changes the scores completely.
#: The analogy to ``KnnScorer.score_k_averaged`` does **not** hold: that
#: computes a well-defined quantity that simply is not reported, whereas this
#: one computes nothing well-defined at all. "Five, not six" is therefore
#: mathematically forced rather than a reporting choice.
#:
#: Tiers declared 2026-08-31. The two unnormalized
#: components are **secondary**: this family's headline claim is the paired
#: difference of ``pca_residual_all_id`` against the naive marginal-diagonal
#: cell, which is a primary claim carried by a secondary-tier scorer -- the tier
#: is about narrative weight, not about which contrasts are pre-specified. The
#: two L2 variants are **appendix**, the other half of the normalization
#: ablation. The whitened variant is **conditional**: it runs either way and is
#: narrated only if rho fires (see ``degeneracy_report``).
CONFIGURATIONS: tuple[PcaResidualConfig, ...] = (
    PcaResidualConfig("pca_residual_all_id", "all_id", reporting_tier="secondary"),
    PcaResidualConfig("pca_residual_class_mean", "class_mean", reporting_tier="secondary"),
    PcaResidualConfig(
        "pca_residual_all_id_l2", "all_id", normalize=True, reporting_tier="appendix"
    ),
    PcaResidualConfig(
        "pca_residual_class_mean_l2",
        "class_mean",
        normalize=True,
        reporting_tier="appendix",
    ),
    PcaResidualConfig(
        "pca_residual_class_mean_whitened",
        "class_mean",
        whiten=True,
        reporting_tier="conditional",
    ),
)

_BY_NAME: dict[str, PcaResidualConfig] = {c.name: c for c in CONFIGURATIONS}


def config_by_name(name: str) -> PcaResidualConfig:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown PCA-residual configuration {name!r}; the reported ones are "
            f"{sorted(_BY_NAME)}"
        ) from None


# --------------------------------------------------------------------------- #
# The scorer
# --------------------------------------------------------------------------- #


class PcaResidualScorer:
    """A fitted subspace. ``score(x)`` returns ``r_d(x)``, higher = more OOD.

    Satisfies ``xai_ood.methods.dump.Scorer`` (a ``name`` and a
    ``score(x) -> (n,)``), so it drops straight into
    ``xai_ood.methods.dump.score_frame`` alongside the Gaussian and kNN
    scorers and a mixed dump is aligned for free.

    ``MANIFEST_REQUIRES`` declares that a run manifest carrying this scorer
    must also carry a ``degeneracy`` fragment.
    ``xai_ood.methods.manifest.run_manifest`` enforces whatever a scorer
    declares without knowing what any particular family needs -- the same
    inversion as ``diagnostics``, where a family joins the manifest by
    implementing something rather than by editing ``manifest.py``.

    Note it deliberately does **not** expose an attribute called ``primary``.
    ``xai_ood.methods.manifest.covariance_manifest`` filters on
    ``isinstance(getattr(s, "primary", None), CovarianceFit)`` and then reads
    ``scorer.background``; a subspace fit named ``primary`` would pass that
    filter and die on the next line. The whitening fit, when there is one, is
    ``whitening_fit``, and its diagnostics reach the manifest through
    ``diagnostics`` like everything else.
    """

    #: Keys a run manifest carrying this scorer must also carry, enforced by
    #: ``xai_ood.methods.manifest.run_manifest``.
    #:
    #: ``degeneracy`` is ``degeneracy_report``'s fragment. It cannot be
    #: collected by ``diagnostics`` like everything else, because rho is a
    #: function of the scorer **and** of the embeddings it was computed on, and
    #: ``run_manifest`` receives scorers only. Declaring it here is what makes
    #: the omission a ``ValueError`` at the call site rather than a manifest
    #: that looks complete and silently is not.
    #:
    #: Rho is the most expensive of the three runtime-measured quantities to
    #: lose.
    #: Without it the pre-declared threshold cannot be applied to the run that
    #: actually happened, and recomputing it later against a possibly different
    #: scored set is precisely the forking path the pre-declaration closes.
    MANIFEST_REQUIRES: tuple[str, ...] = ("degeneracy",)

    def __init__(
        self,
        config: PcaResidualConfig,
        subspace: SubspaceFit,
        d: int,
        *,
        mean: np.ndarray,
        whitening_fit: CovarianceFit | None = None,
    ) -> None:
        if config.whiten and whitening_fit is None:
            raise ValueError(f"{config.name}: a whitened variant needs a whitening fit")
        if not config.whiten and whitening_fit is not None:
            raise ValueError(
                f"{config.name}: an unwhitened variant was given a whitening fit, "
                f"which it would silently ignore."
            )
        self.config = config
        self.subspace = subspace
        self.mean = np.asarray(mean, dtype=np.float64)
        self.whitening_fit = whitening_fit
        self.d = self._check_d(int(d))

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def n_features(self) -> int:
        return self.subspace.n_features

    def _check_d(self, d: int) -> int:
        if d < 1:
            raise ValueError(f"{self.name}: d must be at least 1, got {d}")
        if d > self.n_features:
            raise ValueError(
                f"{self.name}: d = {d} exceeds the embedding dimension "
                f"{self.n_features}. There is no such subspace; the residual "
                f"would be over a negative number of directions."
            )
        return d

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        """Everything before projection: normalize, centre on mu_ID, whiten.

        Returns the **already-centred** vectors the projector acts on, so
        ``score`` never re-centres and the whitened and unwhitened paths
        cannot disagree about where the origin is.
        """
        z = np.asarray(x, dtype=np.float64)
        if z.ndim == 1:
            z = z[None, :]
        if z.shape[1] != self.n_features:
            raise ValueError(
                f"{self.name}: input has {z.shape[1]} dimensions, the fitted "
                f"subspace has {self.n_features}"
            )
        if self.config.normalize:
            z = l2_normalize(z)
        if self.whitening_fit is not None:
            # whiten() subtracts the fit's own mean, which is mu_ID in the same
            # (possibly normalized) space, so this centres and whitens at once.
            return self.whitening_fit.whiten(z)
        return z - self.mean

    def score(self, x: np.ndarray, *, d: int | None = None) -> np.ndarray:
        """``r_d(z) = || (I - P_d) (z - mu_ID) ||_2``. Higher = more OOD.

        ``d`` defaults to the fitted value -- the 95%-rule integer for the all-ID
        subspace, ``K - 1`` for the class-mean one. It is exposed for the same
        reason ``KnnScorer.score`` exposes ``k``: so the appendix sweep calls
        **this** function rather than a second implementation. The headline
        table uses the fitted default regardless of what the sweep shows.

        The residual is formed **directly**, ``e = z_c - V_d (V_d^T z_c)``,
        rather than as ``sqrt(||z_c||^2 - ||P_d z_c||^2)``. The two are equal in
        exact arithmetic; the second is a difference of two large nearly-equal
        numbers whenever the residual is small, which is precisely the regime
        the upper end of ``SWEEP_D`` lives in.
        """
        d = self.d if d is None else self._check_d(int(d))
        centred = self._prepare(x)
        basis = self.subspace.components[:, :d]
        residual = centred - (centred @ basis) @ basis.T
        return np.sqrt(np.einsum("ij,ij->i", residual, residual))

    def score_sweep(
        self,
        x: np.ndarray,
        ds: Sequence[int] = SWEEP_D,
        *,
        block: int = _QUERY_BLOCK,
    ) -> dict[int, np.ndarray]:
        """Residuals for several d from **one** projection pass.

        The appendix figure needs eight values of d; projecting eight times
        would repeat the same ``(n, p) @ (p, p)`` product eight times. With the
        full coefficient vector ``c = V^T z_c`` in hand, every d falls out of a
        reverse cumulative sum::

            r_d^2 = sum_{j > d} c_j^2

        which is a sum of **non-negative** terms, so unlike the analogous
        shortcut in ``score`` it cannot suffer cancellation. It agrees with
        ``score`` to floating-point tolerance rather than bitwise, since the
        two reach the same quantity by different routes;
        ``test_sweep_agrees_with_individual_scores`` bounds the gap.

        **Reported, never used to reselect d.** See ``SWEEP_D``.
        """
        chosen = sorted({int(v) for v in ds})
        for value in chosen:
            self._check_d(value)
        if not chosen:
            raise ValueError(f"{self.name}: no dimensions to sweep")

        components = self.subspace.components
        parts: dict[int, list[np.ndarray]] = {d: [] for d in chosen}

        z = np.asarray(x, dtype=np.float64)
        if z.ndim == 1:
            z = z[None, :]
        for start in range(0, z.shape[0], block):
            centred = self._prepare(z[start : start + block])
            coefficients = centred @ components
            np.square(coefficients, out=coefficients)
            # Tail sums: reverse-cumulate, so tail[i] = sum of c_j^2 for j >= i.
            tail = np.cumsum(coefficients[:, ::-1], axis=1)[:, ::-1]
            for d in chosen:
                # Everything strictly beyond the retained d components. At
                # d == p the retained subspace is the whole space and the
                # residual is exactly zero, by construction rather than by
                # cancellation.
                block_tail = tail[:, d] if d < tail.shape[1] else np.zeros(tail.shape[0])
                parts[d].append(np.sqrt(np.clip(block_tail, 0.0, None)))

        return {
            d: np.concatenate(parts[d]) if parts[d] else np.empty(0, dtype=np.float64)
            for d in chosen
        }

    def diagnostics(self) -> dict[str, Any]:
        """Manifest fragment. Every value is read off the fit at runtime.

        **The integer d goes in here**, which is the whole point: it can only
        be measured once the embeddings exist, so the manifest is where its
        actual value is recorded rather than a source file.
        ``xai_ood.methods.manifest.run_manifest`` collects this
        automatically by calling ``diagnostics()`` on every scorer, so nothing
        in ``manifest.py`` knows about this family.
        """
        spectrum = self.subspace.eigenvalues
        ratio = explained_variance_ratio(spectrum)
        out: dict[str, Any] = {
            "scorer": self.name,
            "hyperparams": self.config.encoded_hyperparams(d=self.d),
            "reporting_tier": self.config.reporting_tier,
            "subspace": self.config.subspace,
            "d": int(self.d),
            "d_rule": (
                f"explained_variance_{VARIANCE_THRESHOLD:g}"
                if self.config.subspace == "all_id"
                else "k_minus_1"
            ),
            "n_features": int(self.n_features),
            "n_residual_directions": int(self.n_features - self.d),
            "n_components_available": int(self.subspace.n_components_available),
            "n_fitting_rows": int(self.subspace.n_fitting_rows),
            "explained_variance_at_d": float(ratio[self.d - 1]),
            "eigenvalue_max": float(spectrum[0]),
            "eigenvalue_min": float(spectrum[-1]),
            "normalize": bool(self.config.normalize),
            "whiten": bool(self.config.whiten),
            "fitting_split": "cifar10_train",
            "centre": "mu_id",
        }
        if self.config.subspace == "all_id":
            # The sweep grid is recorded so the appendix figure and the manifest
            # cannot disagree about what was swept, and so the never-reselect
            # claim is checkable against what actually ran.
            out["sweep_d"] = list(SWEEP_D)
            out["sweep_is_reported_never_reselected"] = True
        if self.whitening_fit is not None:
            out["whitening_covariance"] = dict(self.whitening_fit.diagnostics)
        return out


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def fit_pca_residual(
    config: PcaResidualConfig | str,
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    variance_threshold: float = VARIANCE_THRESHOLD,
) -> PcaResidualScorer:
    """Fit one configuration on ID **train** embeddings. Fit on train, always.

    ``labels`` are required for the ``class_mean`` subspace and ignored for
    ``all_id``. There is deliberately **no ``d`` parameter**: for ``all_id`` the
    integer comes from ``select_dimension_by_variance`` applied to the
    eigenspectrum computed here, and for ``class_mean`` it is ``K - 1``. A
    ``d`` argument would be a place for a guessed value to enter, and the
    never-reselect rule is easier to keep when there is nowhere to put one.
    ``PcaResidualScorer.score(x, d=...)`` covers the appendix sweep's needs.

    ``variance_threshold`` is exposed only so the rule itself can be tested at
    other thresholds; the declared value is ``VARIANCE_THRESHOLD``.
    """
    if isinstance(config, str):
        config = config_by_name(config)

    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{config.name}: expected a 2-D (n, p) array, got shape {x.shape}")
    if config.normalize:
        x = l2_normalize(x)

    mu_id = x.mean(axis=0)

    whitening_fit = None
    space = x
    if config.whiten:
        # Whitening is defined by the marginal, full, unshrunk ID-train
        # covariance in the same space the scorer operates in. Marginal because
        # the remedy is about the ID cloud's overall anisotropy, not about class
        # structure -- using a class-conditional covariance would fold part of
        # the very signal the subspace is meant to carry into the whitening.
        # Unshrunk because shrinkage is an independent factor of the Gaussian
        # family's design and crossing it in here would confound two changes.
        whitening_fit = fit_covariance(
            x,
            None,
            diagonal=False,
            shrinkage=False,
            l2_normalize_first=False,  # already applied above if configured
            name=f"{config.name}:whitening",
        )
        space = whitening_fit.whiten(x)

    if config.class_conditional:
        if labels is None:
            raise ValueError(
                f"{config.name}: labels are required for the class-mean subspace. "
                f"Passing None would fit PCA to the ID cloud instead, silently "
                f"collapsing the two subspaces into one."
            )
        lab = np.asarray(labels)
        if lab.shape[0] != space.shape[0]:
            raise ValueError(
                f"{config.name}: {lab.shape[0]} labels for {space.shape[0]} embeddings"
            )
        classes = np.unique(lab)
        n_classes = int(classes.size)
        if n_classes < 2:
            raise ValueError(
                f"{config.name}: the class-mean subspace needs at least 2 classes, "
                f"got {n_classes}. With one class the centred class-mean matrix "
                f"has rank 0 and there is no subspace."
            )
        rows = np.stack([space[lab == k].mean(axis=0) for k in classes])
        # d = K - 1, every available component. The rank of the centred
        # class-mean matrix is at most K - 1 for any class balance, since
        # sum_k n_k (mu_k - mu_ID) = 0 by the definition of mu_ID.
        d = n_classes - 1
    else:
        classes = None
        n_classes = 0
        rows = space
        d = None  # read off the eigenspectrum below

    # The fitting data is centred on mu_ID in the projection space. Unwhitened,
    # that is mu_ID itself; whitened, the whitening has already subtracted it,
    # so the origin is exactly zero and centring again would be a no-op that
    # only invites the two paths to drift.
    origin = np.zeros(space.shape[1]) if config.whiten else mu_id
    eigenvalues, eigenvectors, rank = _pca_basis(rows, origin, name=config.name)

    if d is None:
        d = select_dimension_by_variance(eigenvalues, threshold=variance_threshold)

    subspace = SubspaceFit(
        origin=origin,
        components=eigenvectors,
        eigenvalues=eigenvalues,
        n_components_available=rank,
        n_fitting_rows=int(rows.shape[0]),
        name=f"{config.name}:subspace",
        diagnostics={
            "name": f"{config.name}:subspace",
            "subspace": config.subspace,
            "n_fitting_rows": int(rows.shape[0]),
            "n_features": int(space.shape[1]),
            "n_components_available": int(rank),
            "n_classes": n_classes,
            "variance_threshold": float(variance_threshold)
            if config.subspace == "all_id"
            else None,
        },
    )
    return PcaResidualScorer(
        config, subspace, d, mean=mu_id, whitening_fit=whitening_fit
    )


def fit_all_pca_residual(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    variance_threshold: float = VARIANCE_THRESHOLD,
    configs: Iterable[PcaResidualConfig] | None = None,
) -> dict[str, PcaResidualScorer]:
    """Fit every reported configuration from one pass over the ID train embeddings."""
    chosen = tuple(configs) if configs is not None else CONFIGURATIONS
    return {
        c.name: fit_pca_residual(c, embeddings, labels, variance_threshold=variance_threshold)
        for c in chosen
    }


# --------------------------------------------------------------------------- #
# Baseline 1: the degeneracy control
# --------------------------------------------------------------------------- #


def degeneracy_report(
    scorer: PcaResidualScorer,
    embeddings: np.ndarray,
    *,
    threshold: float = DEGENERACY_RHO_THRESHOLD,
) -> dict[str, Any]:
    """Is the residual a subspace score, or a rescaled embedding norm?

    With K = 10 classes the class-mean subspace has rank at most 9 out of 768,
    so the residual to it is *nearly the entire centred vector* and may amount
    to little more than ``||z - mu_ID||``. This reports Spearman rho against
    both norms and applies the threshold.

    **The returned rho is computed, never remembered.** It can only be measured
    once the embeddings exist, so this is a function that returns a number rather
    than a number. The threshold beside it *is* a constant, because it was
    declared in advance and that is exactly what makes the verdict non-post-hoc.

    Both norms are taken on the **raw input embeddings as passed**. That is
    deliberate for the L2-normalized variant too: the question is
    whether the residual reduces to the magnitude information present in the
    embedding, and after normalization ``||z||`` is 1 for every row and carries
    none. Correlating against the normalized norm would compare the residual to
    a constant, which ``xai_ood.methods.correlation.pearson_r`` refuses.

    The verdict applies the threshold to the **centred** norm, the sharper of
    the two controls, since the residual is itself computed on centred vectors.
    A high correlation against the plain norm with a low one against the centred
    norm would be the weaker finding and is reported but not decisive.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"degeneracy_report: expected a 2-D (n, p) array, got {z.shape}")

    residual = scorer.score(z)
    plain_norm = np.linalg.norm(z, axis=1)
    centred_norm = np.linalg.norm(z - scorer.mean, axis=1)

    rho_plain = spearman_rho(residual, plain_norm, name=f"{scorer.name}:rho_plain_norm")
    rho_centred = spearman_rho(residual, centred_norm, name=f"{scorer.name}:rho_centred_norm")
    degenerate = bool(rho_centred >= threshold)

    return {
        "scorer": scorer.name,
        "subspace": scorer.config.subspace,
        "d": int(scorer.d),
        "n_features": int(scorer.n_features),
        "n_residual_directions": int(scorer.n_features - scorer.d),
        "n_scored": int(z.shape[0]),
        "rho_plain_norm": rho_plain,
        "rho_centred_norm": rho_centred,
        "threshold": float(threshold),
        "threshold_applies_to": "rho_centred_norm",
        "degenerate": degenerate,
        "verdict": (
            "rescaled_norm_score" if degenerate else "subspace_structure"
        ),
        "remedy": "whitening_before_projection" if degenerate else None,
        "note": (
            "Pre-declared: rho >= threshold against the centred norm means this "
            "component is written up as a rescaled-norm score, not as subspace "
            "structure. The remedy is the separately labeled whitened variant "
            "(pca_residual_class_mean_whitened), never a silent substitution."
        ),
    }
