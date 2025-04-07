import pytest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from counterfactualEO.functions_evaluation import (
    ci_prob,
    ci_prob_diff,
    est_risk_post,
    est_predictive_change,
    est_cFPR_post,
    est_cFNR_post,
    metrics_post_simple,
    metrics_post_crossfit,
    coverage
)


## Testing ci_prob function

def test_ci_prob_expit():
    lower, upper = ci_prob(est=0.6, sd=0.1, z=1.96, n=100, scale='expit')
    assert 0 <= lower <= upper <= 1
    assert np.isclose(upper - lower, 2 * 1.96 * 0.1 / np.sqrt(100), atol=1e-4)

def test_ci_prob_logit_center():
    lower, upper = ci_prob(est=0.5, sd=0.1, z=1.96, n=100, scale='logit')
    assert 0 <= lower < 0.5 < upper <= 1

def test_ci_prob_logit_matches_expit_center():
    # For est=0.5 and small sd, the logit CI should be symmetric around 0.5
    l_logit, u_logit = ci_prob(est=0.5, sd=0.05, z=1.96, n=100, scale='logit')
    assert np.isclose(u_logit - 0.5, 0.5 - l_logit, atol=1e-5)

def test_ci_prob_extreme_estimates():
    for est in [0.01, 0.99]:
        lower, upper = ci_prob(est=est, sd=0.05, z=1.96, n=100, scale='logit')
        assert 0 <= lower <= upper <= 1

def test_ci_prob_monotonicity():
    # CI should be wider for higher z-score
    l1, u1 = ci_prob(0.5, 0.1, z=1.0, n=100, scale='expit')
    l2, u2 = ci_prob(0.5, 0.1, z=2.0, n=100, scale='expit')
    assert (u2 - l2) > (u1 - l1)


## Testing ci_prob_diff function

def test_ci_prob_diff_expit():
    lower, upper = ci_prob_diff(est=0.2, sd=0.1, z=1.96, n=100, scale='expit')
    assert -1 <= lower <= upper <= 1
    assert np.isclose(upper - lower, 2 * 1.96 * 0.1 / np.sqrt(100), atol=1e-4)

def test_ci_prob_diff_logit_center():
    lower, upper = ci_prob_diff(est=0.0, sd=0.1, z=1.96, n=100, scale='logit')
    assert -1 <= lower < 0 < upper <= 1

def test_ci_prob_diff_logit_matches_expit_center():
    l_logit, u_logit = ci_prob_diff(est=0.0, sd=0.05, z=1.96, n=100, scale='logit')
    assert np.isclose(u_logit - 0.0, 0.0 - l_logit, atol=1e-5)

def test_ci_prob_diff_extreme_estimates():
    for est in [-0.98, 0.98]:
        lower, upper = ci_prob_diff(est=est, sd=0.05, z=1.96, n=100, scale='logit')
        assert -1 <= lower <= upper <= 1

def test_ci_prob_diff_monotonicity():
    l1, u1 = ci_prob_diff(0.3, 0.1, z=1.0, n=100, scale='expit')
    l2, u2 = ci_prob_diff(0.3, 0.1, z=2.0, n=100, scale='expit')
    assert (u2 - l2) > (u1 - l1)


@pytest.mark.parametrize("ci_scale", ['expit', 'logit'])
def test_est_risk(real_data, ci_scale):
    df, testdict = real_data
    theta = testdict['theta']
    result = est_risk_post(theta, df, outcome='phihat', ci=0.95, ci_scale=ci_scale)
    expected = testdict['risk_df'] if ci_scale == 'expit' else testdict['risk_df_logit']
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ci_scale", ['expit', 'logit'])
def test_est_cFPR_post(real_data, ci_scale):
    df, testdict = real_data
    theta = testdict['theta']
    result = est_cFPR_post(theta, df, outcome='phihat', ci=0.95, ci_scale=ci_scale)
    expected = testdict['cFPR_df'] if ci_scale == 'expit' else testdict['cFPR_df_logit']
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ci_scale", ['expit', 'logit'])
def test_est_cFNR_post(real_data, ci_scale):
    df, testdict = real_data
    theta = testdict['theta']
    result = est_cFNR_post(theta, df, outcome='phihat', ci=0.95, ci_scale=ci_scale)
    expected = testdict['cFNR_df'] if ci_scale == 'expit' else testdict['cFNR_df_logit']
    pd.testing.assert_frame_equal(result, expected)


def test_metrics_post_simple_structure(real_data):
    df, testdict = real_data
    theta = testdict['theta']
    result = metrics_post_simple(theta, df, outcome='phihat', ci=0.95)
    assert isinstance(result, pd.DataFrame)
    assert {'metric', 'value', 'ci_lower', 'ci_upper'}.issubset(result.columns)


def test_metrics_post_crossfit_output(mock_data):
    theta = np.array([0.2, 0.3, 0.4, 0.5])
    X = ['X1', 'X2']
    model = LogisticRegression()

    result = metrics_post_crossfit(
        theta=theta,
        data=mock_data,
        A='A',
        X=X,
        R='R',
        D='D',
        Y='Y',
        learner_pi=model,
        learner_mu=model,
        outcome='phihat',
        ci=0.95,
        ci_scale='logit',
        n_splits=2,
        random_state=42
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) >= {'metric', 'value', 'ci_lower', 'ci_upper'}

    expected_metrics = {
        'risk', 'risk_change',
        'FPR0', 'FPR1', 'gap_FPR',
        'FNR0', 'FNR1', 'gap_FNR',
        'pred_change'
    }
    assert set(result['metric']) == expected_metrics

    # Metrics that can lie in [-1, 1]
    unbounded_metrics = {'risk_change', 'pred_change', 'gap_FPR', 'gap_FNR'}

    # Range checks based on metric type
    for _, row in result.iterrows():
        metric = row['metric']
        for col in ['value', 'ci_lower', 'ci_upper']:
            val = row[col]
            if pd.isna(val):
                continue
            if metric in unbounded_metrics:
                assert -1 <= val <= 1, f"{metric} - {col} out of [-1, 1]: {val}"
            else:
                assert 0 <= val <= 1, f"{metric} - {col} out of [0, 1]: {val}"


def test_metrics_post_crossfit_invalid_n_splits(mock_data):
    theta = np.array([0.2, 0.3, 0.4, 0.5])
    X = ['X1', 'X2']
    model = LogisticRegression()

    with pytest.raises(ValueError):
        metrics_post_crossfit(
            theta=theta,
            data=mock_data,
            A='A',
            X=X,
            R='R',
            D='D',
            Y='Y',
            learner_pi=model,
            learner_mu=model,
            n_splits=1,  # invalid, must be at least 2
        )


def test_coverage_structure(real_data):
    df, testdict = real_data
    metrics_est = metrics_post_simple(testdict['theta'], df, outcome='phihat', ci=0.95)
    cov_df = coverage(metrics_est, testdict['risk_df'], simplify=False)
    assert isinstance(cov_df, pd.DataFrame)


def test_coverage_with_simplify(mock_metrics_est_with_n, mock_metrics_true):
    """
    Ensure coverage runs and returns expected output when simplify=True and 'n' is present.
    """
    result = coverage(mock_metrics_est_with_n, mock_metrics_true, simplify=True)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ['metric', 'n', 'coverage']
    assert result.shape[0] > 0

@pytest.mark.parametrize("ci", [0.95, None])
def test_est_predictive_change_output_structure(mock_data, ci):
    theta = np.array([0.2, 0.3, 0.4, 0.5])
    result = est_predictive_change(theta, mock_data, A='A', R='R', ci=ci)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 4)
    assert set(result.columns) == {'metric', 'value', 'ci_lower', 'ci_upper'}
    assert result.loc[0, 'metric'] == 'pred_change'

def test_est_predictive_change_values_in_range(mock_data):
    theta = np.array([0.2, 0.3, 0.4, 0.5])
    result = est_predictive_change(theta, mock_data, A='A', R='R', ci=0.95)

    val = result.loc[0, 'value']
    assert 0 <= val <= 1
    assert 0 <= result.loc[0, 'ci_lower'] <= 1
    assert 0 <= result.loc[0, 'ci_upper'] <= 1

def test_est_predictive_change_ci_none(mock_data):
    theta = np.array([0.5, 0.5, 0.5, 0.5])
    result = est_predictive_change(theta, mock_data, A='A', R='R', ci=None)

    assert pd.isna(result.loc[0, 'ci_lower'])
    assert pd.isna(result.loc[0, 'ci_upper'])

def test_est_predictive_change_ci_band_order(mock_data):
    theta = np.array([0.7, 0.3, 0.6, 0.4])
    result = est_predictive_change(theta, mock_data, A='A', R='R', ci=0.95)

    lower = result.loc[0, 'ci_lower']
    upper = result.loc[0, 'ci_upper']
    assert lower <= upper

def test_est_predictive_change_value_changes_with_theta(mock_data):
    theta1 = np.array([0.1, 0.2, 0.3, 0.4])
    theta2 = np.array([0.9, 0.8, 0.7, 0.6])
    result1 = est_predictive_change(theta1, mock_data, A='A', R='R', ci=None)
    result2 = est_predictive_change(theta2, mock_data, A='A', R='R', ci=None)

    val1 = result1.loc[0, 'value']
    val2 = result2.loc[0, 'value']
    assert not np.isclose(val1, val2), "Predictive change should differ across thetas"