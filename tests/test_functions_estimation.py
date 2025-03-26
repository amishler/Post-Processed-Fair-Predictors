import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression

from counterfactualEO import functions_estimation as fe

#############################################
#### Fixtures for synthetic & saved data ####
#############################################

@pytest.fixture
def mock_data():
    n = 100
    return pd.DataFrame({
        'A': np.random.randint(0, 2, size=n),
        'X1': np.random.randn(n),
        'X2': np.random.randn(n),
        'R': np.random.randint(0, 2, size=n),
        'D': np.random.randint(0, 2, size=n),
        'Y': np.random.randint(0, 2, size=n)
    })


##############################
#### Tests on mock data  #####
##############################

def test_train_nuisance(mock_data):
    X = ['X1', 'X2']
    model = LogisticRegression()
    out = fe.train_nuisance(mock_data.iloc[:50], mock_data.iloc[50:], A='A', X=X, R='R', D='D', Y='Y',
                            learner_pi=model, learner_mu=model)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {'pihat', 'muhat0', 'phihat'}


def test_risk_coef_and_risk_coefs(mock_data):
    mock_data['phihat'] = np.random.rand(len(mock_data))
    single = fe.risk_coef(mock_data, 0, 1, outcome='phihat')
    assert isinstance(single, float)

    coefs = fe.risk_coefs(mock_data, outcome='phihat')
    assert isinstance(coefs, np.ndarray)
    assert coefs.shape == (4,)


def test_cFPR_and_cFNR(mock_data):
    mock_data['phihat'] = np.clip(np.random.rand(len(mock_data)), 0.01, 0.99)
    fpr = fe.cFPR(mock_data, R='R', outcome='phihat')
    fnr = fe.cFNR(mock_data, R='R', outcome='phihat')
    assert 0 <= fpr <= 1
    assert 0 <= fnr <= 1


def test_fairness_coefs(mock_data):
    mock_data['phihat'] = np.clip(np.random.rand(len(mock_data)), 0.01, 0.99)
    pos, neg = fe.fairness_coefs(mock_data, A='A', R='R', outcome='phihat')
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (4,)
    assert isinstance(neg, np.ndarray)
    assert neg.shape == (4,)


def test_optimize():
    risk = np.array([0.1, 0.2, 0.3, 0.4])
    coefs_pos = np.array([1, -1, 0, 0])
    coefs_neg = np.array([0, 0, 1, -1])
    theta = fe.optimize(risk, coefs_pos, coefs_neg, epsilon_pos=0.1, epsilon_neg=0.1)
    assert isinstance(theta, np.ndarray)
    assert theta.shape == (4,)
    assert np.all((0 <= theta) & (theta <= 1))


def test_fair_derived(mock_data):
    X = ['X1', 'X2']
    model = LogisticRegression()
    result = fe.fair_derived(data=mock_data, A='A', X=X, R='R', D='D', Y='Y',
                             learner_pi=model, learner_mu=model,
                             epsilon_pos=0.1, epsilon_neg=0.1)
    assert isinstance(result, dict)
    assert 'theta' in result
    assert result['theta'].shape == (4,)


########################################
#### Tests using real reference data ###
########################################

@pytest.mark.parametrize("outcome_key", ['phihat', 'mu0'])
def test_risk_coefs_real_data(real_data, outcome_key):
    df, testdict = real_data
    result = fe.risk_coefs(df, A='A', R='R', outcome=outcome_key)
    expected = testdict[f'risk_coefs_{outcome_key}']
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("outcome_key", ['phihat', 'mu0'])
def test_fairness_coefs_real_data(real_data, outcome_key):
    df, testdict = real_data
    pos, neg = fe.fairness_coefs(df, outcome='phihat' if outcome_key == 'phihat' else 'mu0')
    np.testing.assert_allclose(pos, testdict[f'coefs_pos_{outcome_key}'])
    np.testing.assert_allclose(neg, testdict[f'coefs_neg_{outcome_key}'])

@pytest.mark.parametrize("ci_scale", ['expit', 'logit'])
def test_est_risk(real_data, ci_scale):
    df, testdict = real_data
    theta = testdict['theta']
    result = fe.est_risk(theta, df, outcome='phihat', ci=0.95, ci_scale=ci_scale)
    expected = testdict['risk_df'] if ci_scale == 'expit' else testdict['risk_df_logit']
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ci_scale", ['expit', 'logit'])
def test_est_cFPR(real_data, ci_scale):
    df, testdict = real_data
    theta = testdict['theta']
    result = fe.est_cFPR(theta, df, outcome='phihat', ci=0.95, ci_scale=ci_scale)
    expected = testdict['cFPR_df'] if ci_scale == 'expit' else testdict['cFPR_df_logit']
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("ci_scale", ['expit', 'logit'])
def test_est_cFNR(real_data, ci_scale):
    df, testdict = real_data
    theta = testdict['theta']
    result = fe.est_cFNR(theta, df, outcome='phihat', ci=0.95, ci_scale=ci_scale)
    expected = testdict['cFNR_df'] if ci_scale == 'expit' else testdict['cFNR_df_logit']
    pd.testing.assert_frame_equal(result, expected)