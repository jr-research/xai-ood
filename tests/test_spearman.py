# tests/test_spearman.py
#
# Guards xai_ood.methods.correlation, the numpy Spearman the PCA-residual
# degeneracy control needs.
#
# The obvious call is scipy.stats.spearmanr, and tests/test_import_hygiene.py
# forbids scipy in src/. This is the third time that rule has cost something --
# Ledoit-Wolf, and exact search instead of faiss for kNN -- and
# the trade is the same each time: the algorithm is unchanged, a dependency-free
# test checks it against arithmetic that shares no code with the implementation,
# and a cross-check against the real library runs wherever the library exists.
#
# Every expectation below is either hand-computed in the docstring or a closed
# form (exactly +/- 1). Nothing is a value remembered from a previous run.
import numpy as np
import pytest

from xai_ood.methods.correlation import average_ranks, pearson_r, spearman_rho


# --------------------------------------------------------------------------- #
# Ranks
# --------------------------------------------------------------------------- #


def test_ranks_of_distinct_values_are_the_sorted_positions():
    """[30, 10, 20] -> [3, 1, 2]. One-based, ascending."""
    assert average_ranks([30.0, 10.0, 20.0]).tolist() == [3.0, 1.0, 2.0]


def test_tied_values_share_their_average_rank():
    """[10, 20, 20, 40] occupies positions 1..4.

    The two 20s occupy positions 2 and 3, so both get (2 + 3) / 2 = 2.5.
    """
    assert average_ranks([10.0, 20.0, 20.0, 40.0]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_a_three_way_tie_averages_all_three_positions():
    """[5, 5, 5, 9]: positions 1, 2, 3 average to 2.0; the 9 takes position 4."""
    assert average_ranks([5.0, 5.0, 5.0, 9.0]).tolist() == [2.0, 2.0, 2.0, 4.0]


def test_all_values_tied_gives_every_row_the_same_rank():
    """Four equal values occupy positions 1..4, averaging to 2.5 each.

    This is the input that makes the correlation undefined downstream, which is
    tested separately; ranking it is perfectly well defined.
    """
    assert average_ranks([7.0, 7.0, 7.0, 7.0]).tolist() == [2.5] * 4


def test_ranks_always_sum_to_n_times_n_plus_one_over_two():
    """Average-rank tie handling preserves the total, whatever the ties are.

    Sum of 1..n is n(n+1)/2 and averaging within a tie group redistributes
    positions without changing their sum. A tie-breaking scheme that dropped or
    duplicated a position would fail this at some n.
    """
    rng = np.random.default_rng(11)
    for n in (2, 5, 17, 64):
        # Coarse rounding to force plenty of ties.
        values = np.round(rng.normal(size=n), 1)
        assert average_ranks(values).sum() == pytest.approx(n * (n + 1) / 2)


def test_ranks_reject_nan_rather_than_sorting_it_to_the_end():
    """numpy sorts NaN last without complaint, which would silently rank it top."""
    with pytest.raises(ValueError, match="NaN or infinity"):
        average_ranks([1.0, np.nan, 3.0])


def test_ranks_reject_an_empty_input():
    with pytest.raises(ValueError, match="empty input"):
        average_ranks([])


# --------------------------------------------------------------------------- #
# Pearson
# --------------------------------------------------------------------------- #


def test_pearson_of_a_perfect_line_is_one():
    assert pearson_r([1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 9.0]) == pytest.approx(1.0)


def test_pearson_of_a_perfect_negative_line_is_minus_one():
    assert pearson_r([1.0, 2.0, 3.0, 4.0], [9.0, 7.0, 5.0, 3.0]) == pytest.approx(-1.0)


def test_pearson_hand_computed():
    """a = [1, 2, 3, 4], b = [2, 4, 5, 4].

    mean(a) = 2.5, mean(b) = 3.75.
    da = [-1.5, -0.5, 0.5, 1.5];  db = [-1.75, 0.25, 1.25, 0.25].
    sum(da * db) = 2.625 - 0.125 + 0.625 + 0.375 = 3.5
    ||da|| = sqrt(2.25 + 0.25 + 0.25 + 2.25) = sqrt(5)
    ||db|| = sqrt(3.0625 + 0.0625 + 1.5625 + 0.0625) = sqrt(4.75)
    r = 3.5 / sqrt(23.75) = 0.718184...
    """
    assert pearson_r([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 4.0]) == pytest.approx(
        3.5 / np.sqrt(23.75)
    )


def test_pearson_raises_on_a_constant_input_rather_than_returning_nan():
    """scipy returns NaN with a warning here. A NaN correlation propagates.

    A constant score array is not an edge case to absorb: it says the scorer
    separates nothing, which is the finding. Input that is already wrong raises
    at the point it is detected.
    """
    with pytest.raises(ValueError, match="constant"):
        pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="constant"):
        pearson_r([1.0, 2.0, 3.0], [4.0, 4.0, 4.0])


def test_pearson_reports_which_input_was_constant():
    """Both messages exist and differ, so the caller is told which array to look at."""
    with pytest.raises(ValueError, match="first input is constant"):
        pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="second input is constant"):
        pearson_r([1.0, 2.0, 3.0], [4.0, 4.0, 4.0])


def test_pearson_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        pearson_r([1.0, 2.0, 3.0], [1.0, 2.0])


def test_pearson_needs_at_least_two_observations():
    with pytest.raises(ValueError, match="at least 2"):
        pearson_r([1.0], [2.0])


# --------------------------------------------------------------------------- #
# Spearman
# --------------------------------------------------------------------------- #


def test_spearman_of_a_strictly_increasing_pair_is_exactly_one():
    assert spearman_rho([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]) == (
        pytest.approx(1.0)
    )


def test_spearman_of_a_reversed_pair_is_exactly_minus_one():
    assert spearman_rho([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]) == (
        pytest.approx(-1.0)
    )


def test_spearman_is_one_under_a_monotone_nonlinear_map_where_pearson_is_not():
    """b = a**3. Rank correlation 1; Pearson strictly below it.

    This is the property the degeneracy control depends on. "The class-mean
    residual is a rescaled embedding norm" is a claim about *order*: a residual
    that is a nonlinear but order-preserving function of the norm is exactly as
    degenerate as one that is a linear function of it, and a Pearson
    correlation would understate it.
    """
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a**3
    assert spearman_rho(a, b) == pytest.approx(1.0)
    assert pearson_r(a, b) < 0.98


def test_spearman_hand_computed_with_a_tie():
    """a = [1, 2, 2, 3] -> ranks [1, 2.5, 2.5, 4]; b = [10, 20, 30, 40] -> [1, 2, 3, 4].

    Both rank vectors have mean 2.5.
    da = [-1.5, 0, 0, 1.5];   db = [-1.5, -0.5, 0.5, 1.5]
    sum(da * db) = 2.25 + 0 + 0 + 2.25 = 4.5
    ||da|| = sqrt(4.5);  ||db|| = sqrt(5)
    rho = 4.5 / sqrt(22.5) = 3 / sqrt(10) = 0.9486832980505138
    """
    rho = spearman_rho([1.0, 2.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0])
    assert rho == pytest.approx(3.0 / np.sqrt(10.0))


def test_spearman_is_invariant_to_a_shared_permutation_of_both_inputs():
    """Correlation is a property of the pairing, not of the row order.

    The bootstrap applies one index set to every scorer's array, so a
    correlation that depended on row order would make the disagreement analysis
    depend on the sort.
    """
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    b = a * 0.4 + rng.normal(size=200)
    permutation = rng.permutation(200)
    assert spearman_rho(a[permutation], b[permutation]) == pytest.approx(spearman_rho(a, b))


def test_spearman_is_symmetric_in_its_arguments():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=120), rng.normal(size=120)
    assert spearman_rho(a, b) == pytest.approx(spearman_rho(b, a))


def test_spearman_of_a_variable_with_itself_is_one_even_with_ties():
    rng = np.random.default_rng(5)
    a = np.round(rng.normal(size=300), 1)
    assert spearman_rho(a, a) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Cross-check against scipy, on any machine that has it
# --------------------------------------------------------------------------- #


def test_matches_scipy_spearmanr_on_continuous_data():
    """Un-skips wherever scipy is installed.

    Same shape as the scikit-learn cross-check of the Ledoit-Wolf
    reimplementation: importorskip, so a run without scipy skips rather than
    turning red, and the equivalence is asserted automatically wherever scipy
    exists.
    """
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(7)
    for n in (10, 100, 2000):
        a = rng.normal(size=n)
        b = a * 0.6 + rng.normal(size=n)
        assert spearman_rho(a, b) == pytest.approx(
            float(stats.spearmanr(a, b).statistic), abs=1e-12
        )


def test_matches_scipy_spearmanr_with_heavy_ties():
    """The tie path is the half most likely to differ between implementations."""
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(8)
    for n in (12, 250, 1500):
        # Integer-valued, so ties are guaranteed and frequent.
        a = rng.integers(0, 5, size=n).astype(float)
        b = rng.integers(0, 4, size=n).astype(float)
        assert spearman_rho(a, b) == pytest.approx(
            float(stats.spearmanr(a, b).statistic), abs=1e-12
        )


def test_average_ranks_matches_scipy_rankdata():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(9)
    values = np.round(rng.normal(size=500), 1)
    assert np.array_equal(average_ranks(values), stats.rankdata(values, method="average"))
