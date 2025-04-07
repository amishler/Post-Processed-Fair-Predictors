import pytest

import numpy as np
import pandas as pd

from counterfactualEO.functions_evaluation import (
    est_risk_post,
    est_predictive_change,
    est_cFPR_post,
    est_cFNR_post,
    metrics_post,
    coverage
)


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


def test_metrics_post_structure(real_data):
    df, testdict = real_data
    theta = testdict['theta']
    result = metrics_post(theta, df, outcome='phihat', ci=0.95)
    assert isinstance(result, pd.DataFrame)
    assert {'metric', 'value', 'ci_lower', 'ci_upper'}.issubset(result.columns)


def test_coverage_structure(real_data):
    df, testdict = real_data
    metrics_est = metrics_post(testdict['theta'], df, outcome='phihat', ci=0.95)
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
    assert result.loc[0, 'metric'] == 'prop_differ'

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