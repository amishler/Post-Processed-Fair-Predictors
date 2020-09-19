import pytest

import pandas as pd

from counterfactualEO.functions_evaluation import (
    est_risk,
    est_cFPR_group,
    est_cFNR_group,
    metrics,
    coverage
)


@pytest.mark.parametrize("ci_scale", ['expit', 'logit'])
def test_est_risk(real_data, ci_scale):
    df, testdict = real_data
    theta = testdict['theta']
    result = est_risk(theta, df, outcome='phihat', ci=0.95, ci_scale=ci_scale)
    expected = testdict['risk_df'] if ci_scale == 'expit' else testdict['risk_df_logit']
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ci_scale", ['expit', 'logit'])
def test_est_cFPR_group(real_data, ci_scale):
    df, testdict = real_data
    theta = testdict['theta']
    result = est_cFPR_group(theta, df, outcome='phihat', ci=0.95, ci_scale=ci_scale)
    expected = testdict['cFPR_df'] if ci_scale == 'expit' else testdict['cFPR_df_logit']
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ci_scale", ['expit', 'logit'])
def test_est_cFNR_group(real_data, ci_scale):
    df, testdict = real_data
    theta = testdict['theta']
    result = est_cFNR_group(theta, df, outcome='phihat', ci=0.95, ci_scale=ci_scale)
    expected = testdict['cFNR_df'] if ci_scale == 'expit' else testdict['cFNR_df_logit']
    pd.testing.assert_frame_equal(result, expected)


def test_metrics_structure(real_test_data):
    df, testdict = real_test_data
    theta = testdict['theta']
    result = metrics(theta, df, outcome='phihat', ci=0.95)
    assert isinstance(result, pd.DataFrame)
    assert {'metric', 'value', 'ci_lower', 'ci_upper'}.issubset(result.columns)


def test_coverage_structure(real_test_data):
    df, testdict = real_test_data
    metrics_est = metrics(testdict['theta'], df, outcome='phihat', ci=0.95)
    cov_df = coverage(metrics_est, testdict['risk_df'], simplify=True)
    assert isinstance(cov_df, pd.DataFrame)
