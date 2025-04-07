import pytest

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from counterfactualEO.functions_estimation import (
    train_nuisance,
    risk_coef,
    risk_coefs,
    est_cFPR_pre,
    est_cFNR_pre,
    fairness_coefs,
    optimize,
    fair_derived_split,
    fair_derived_crossfit
)


##############################
#### Tests on mock data  #####
##############################

def test_train_nuisance(mock_data):
    X = ['X1', 'X2']
    model = LogisticRegression()
    out = train_nuisance(mock_data.iloc[:50], mock_data.iloc[50:], A='A', X=X, R='R', D='D', Y='Y',
                            learner_pi=model, learner_mu=model)
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {'pihat', 'muhat0', 'phihat'}


def test_risk_coef_and_risk_coefs(mock_data):
    mock_data['phihat'] = np.random.rand(len(mock_data))
    single = risk_coef(mock_data, 0, 1, outcome='phihat')
    assert isinstance(single, float)

    coefs = risk_coefs(mock_data, outcome='phihat')
    assert isinstance(coefs, np.ndarray)
    assert coefs.shape == (4,)


def test_est_cFPR_and_cFNR_overall(mock_data):
    mock_data['phihat'] = np.clip(np.random.rand(len(mock_data)), 0.01, 0.99)
    fpr = est_cFPR_pre(mock_data, R='R', outcome='phihat')
    fnr = est_cFNR_pre(mock_data, R='R', outcome='phihat')
    assert 0 <= fpr <= 1
    assert 0 <= fnr <= 1


def test_fairness_coefs(mock_data):
    mock_data['phihat'] = np.clip(np.random.rand(len(mock_data)), 0.01, 0.99)
    pos, neg = fairness_coefs(mock_data, A='A', R='R', outcome='phihat')
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (4,)
    assert isinstance(neg, np.ndarray)
    assert neg.shape == (4,)


def test_optimize():
    risk = np.array([0.1, 0.2, 0.3, 0.4])
    coefs_pos = np.array([1, -1, 0, 0])
    coefs_neg = np.array([0, 0, 1, -1])
    theta = optimize(risk, coefs_pos, coefs_neg, epsilon_pos=0.1, epsilon_neg=0.1)
    assert isinstance(theta, np.ndarray)
    assert theta.shape == (4,)
    assert np.all((0 <= theta) & (theta <= 1))


def test_fair_derived_split(mock_data):
    X = ['X1', 'X2']
    model = LogisticRegression()
    result = fair_derived_split(data=mock_data, A='A', X=X, R='R', D='D', Y='Y',
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
    result = risk_coefs(df, A='A', R='R', outcome=outcome_key)
    expected = testdict[f'risk_coefs_{outcome_key}']
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("outcome_key", ['phihat', 'mu0'])
def test_fairness_coefs_real_data(real_data, outcome_key):
    df, testdict = real_data
    pos, neg = fairness_coefs(df, outcome='phihat' if outcome_key == 'phihat' else 'mu0')
    np.testing.assert_array_almost_equal(pos, testdict[f'coefs_pos_{outcome_key}'])
    np.testing.assert_array_almost_equal(neg, testdict[f'coefs_neg_{outcome_key}'])


@pytest.mark.parametrize("estimator_type", ['pooled', 'foldwise'])
def test_fair_derived_crossfit(mock_data, estimator_type):
    X = ['X1', 'X2']
    model = LogisticRegression()

    result = fair_derived_crossfit(
        data=mock_data,
        A='A',
        X=X,
        R='R',
        D='D',
        Y='Y',
        learner_pi=model,
        learner_mu=model,
        epsilon_pos=0.1,
        epsilon_neg=0.1,
        estimator_type=estimator_type,
        n_splits=2,
        random_state=42
    )

    assert isinstance(result, dict)
    assert 'theta' in result
    assert result['theta'].shape == (4,)
    assert 'risk_coefs' in result
    assert 'fairness_coefs_pos' in result
    assert 'fairness_coefs_neg' in result
