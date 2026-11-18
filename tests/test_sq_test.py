import numpy as np
import pytest
from scipy import stats

from keyperson.sq_test import reject, sq_test, window_sums


def test_window_sums_one_day_is_a_copy():
    r = np.array([1.0, -2.0, 3.0])
    out = window_sums(r, 1)
    assert np.array_equal(out, r)
    out[0] = 99
    assert r[0] == 1.0


def test_window_sums_overlapping():
    r = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(window_sums(r, 2), [3.0, 5.0, 7.0])
    assert np.array_equal(window_sums(r, 4), [10.0])


def test_window_sums_drops_nan_and_checks_length():
    assert np.array_equal(window_sums(np.array([1.0, np.nan, 2.0]), 2), [3.0])
    with pytest.raises(ValueError):
        window_sums(np.array([1.0]), 2)
    with pytest.raises(ValueError):
        window_sums(np.array([1.0]), 0)


def test_p_values_on_a_known_reference():
    reference = np.arange(1, 101, dtype=float)  # 1..100
    res = sq_test(3.0, reference)
    assert res.p_lower == pytest.approx(4 / 101)  # 1, 2, 3 and the value itself
    assert res.p_upper == pytest.approx(99 / 101)
    assert res.p_two_sided == pytest.approx(8 / 101)
    assert res.reference_size == 100
    assert sq_test(50.5, reference).p_two_sided == pytest.approx(1.0)


def test_value_beyond_every_residual_gets_the_smallest_p():
    res = sq_test(-10.0, np.arange(1, 251, dtype=float))
    assert res.p_lower == pytest.approx(1 / 251) == pytest.approx(res.min_p)
    assert res.p_two_sided == pytest.approx(2 / 251)


def test_symmetric_under_sign_flip():
    rng = np.random.default_rng(1)
    resid = rng.standard_t(4, 250)
    a = sq_test(1.7, resid)
    b = sq_test(-1.7, -resid)
    assert a.p_two_sided == pytest.approx(b.p_two_sided)
    assert a.p_lower == pytest.approx(b.p_upper)


def test_multi_day_reference_is_the_window_sums():
    resid = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    res = sq_test(-0.1, resid, length=2)  # sums: -0.1, 0.1, -0.1, 0.1
    assert res.reference_size == 4
    assert res.p_lower == pytest.approx(3 / 5)


def test_reject_matches_the_quantiles():
    reference = np.arange(1, 201, dtype=float)
    assert reject(0.5, reference, alpha=0.05)
    assert reject(200.5, reference, alpha=0.05)
    assert not reject(100.0, reference, alpha=0.05)
    with pytest.raises(ValueError):
        reject(1.0, reference, alpha=1.5)


def test_size_is_close_to_nominal_with_fat_tails_where_the_normal_test_is_not():
    """Under the null (event return drawn from the residual distribution, t with 3 df), the SQ test
    rejects at most about 1% of the time at the 1% level (exactly 2/251 = 0.8% with 250 residuals); the
    normal test with the sample sd rejects about twice as often, because t(3) tails are fatter."""
    rng = np.random.default_rng(20131)
    reps, n, alpha = 6000, 250, 0.01
    sq_rej = normal_rej = 0
    critical = stats.norm.ppf(1 - alpha / 2)
    for _ in range(reps):
        resid = rng.standard_t(3, n)
        x = rng.standard_t(3)
        sq_rej += sq_test(x, resid).p_two_sided <= alpha
        normal_rej += abs(x / resid.std(ddof=1)) > critical
    assert sq_rej / reps == pytest.approx(2 / 251, abs=0.004)
    assert sq_rej / reps <= alpha + 0.004
    assert normal_rej / reps > 1.5 * alpha


def test_nan_value_gives_nan_p_values():
    res = sq_test(float("nan"), np.arange(10, dtype=float))
    assert np.isnan(res.p_two_sided)
