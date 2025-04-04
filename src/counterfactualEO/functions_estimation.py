"""
functions_estimation.py

This module contains functions to post-process binary predictors to achieve
(approximate) counterfactual equalized odds. It includes:

- Estimation of nuisance parameters (propensity scores and outcome regressions)
- Construction of doubly robust pseudo-outcomes
- Computation of coefficients for linear programs enforcing fairness constraints
- Solving fairness-constrained risk minimization problem to estimate the
    optimal decision variable theta, which indexes the fair predictor.
"""

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_nuisance(train, test, A, X, R, D, Y, learner_pi, learner_mu,
                   trunc_pi=0.975):
    """Train nuisance parameter estimators and return test set predictions."
    
    The nuisance parameters are the propensity model and the outcome regression
    under the baseline treatment level, meaning the outcome regression under 
    D = 0. Models are trained on the training data and evaluated on the test data.
    
    Args:
        train: training data (DataFrame)
        test: test data (DataFrame)
        A: name of the sensitive feature column (str)
        X: names of the feature columns (list)
        R: name of the risk score column (str)
        D: name of the treatment column, aka the decision column (str)
        Y: name of the outcome column (str)
        learner_pi: learner for the propensity score (sklearn model)
        learner_mu: learner for the outcome model (sklearn model)
        trunc_pi: truncation level for the propensity score, a number in (0, 1), 
            default 0.975. Propensity scores are clipped to be at most this value.
            This is useful for preventing numerical instability.
    
    Returns:
        out: DataFrame with columns 'pihat', 'muhat0', 'phihat', where 'pihat'
            represents the estimated propensity scores, 'muhat0' represents the
            estimated outcomes under the baseline treatment (that is, under 
            D = 0), and 'phihat' represents the doubly robust pseudo-outcomes
            under the baseline treatment.
    """
    train = train.reset_index()
    test = test.reset_index()
    # TODO: separate models for the two levels of the sensitive feature A
    # TODO: this doesn't handle categorical features. They need to be transformed beforehand.
    pred_cols = [A] + X + [R]
    learner_pi.fit(train[pred_cols], train[D])
    learner_mu.fit(train.loc[train.D.eq(0), pred_cols],
                   train.loc[train.D.eq(0), Y])
    if hasattr(learner_pi, 'predict_proba'):
        pihat = pd.Series(learner_pi.predict_proba(test[pred_cols])[:, 1],
                          name='pihat').clip(upper=trunc_pi)
    else:
        pihat = pd.Series(learner_pi.predict(test[pred_cols]),
                          name='pihat').clip(upper=trunc_pi)
    if hasattr(learner_mu, 'predict_proba'):
        muhat0 = pd.Series(learner_mu.predict_proba(test[pred_cols])[:, 1],
                           name='muhat0')
    else:
        muhat0 = pd.Series(learner_mu.predict(test[pred_cols]), name='muhat0')
    phihat = pd.Series(
        (1 - test[D]) / (1 - pihat) * (test[Y] - muhat0) + muhat0,
        name='phihat')

    out = pd.concat([pihat, muhat0, phihat], axis=1)

    return out


########################################################
#### Risk functions, for the optimization objective ####
########################################################

def risk_coef(data, a, r, A='A', R='R', outcome='phihat'):
    """Compute the risk coefficient $\beta_{a, r}$ for given values a, r.

    This computes the coefficient for a single optimization variable 
    $\theta_{a, r}$.

    The outcome should be one of 'phihat', 'muhat0', 'mu0'. In general, `data` 
    will be generated by `train_nuisance`, in which case 'phihat' will
    represent doubly robust pseudo-outcomes under the baseline treatment level,
    while muhat0 will represent the outcome regression under the baseline
    treatment level. In this case, using phihat will yield a doubly robust
    estimator of the risk, while using muhat0 will yield a plugin estimator of
    the risk. In general, doubly robust estimators are preferred unless the
    outcome regression model is known to be correctly specified.

    The outcome mu0 is intended to represent oracle knowledge of the outcome
    regression, which is only useful in the context of simulations.

    Args:
        data: DataFrame containing columns A, R, and the outcome. In general,
            this will be the output of train_nuisance, and the outcome will be
            'phihat'.
        a: value of A, the sensitive feature, either 0 or 1.
        r: value of R, the risk prediction, either 0 or 1.
        A: name of the sensitive feature column (str).
        R: name of the risk prediction column (str).
        outcome: name of the outcome column, one of 'phihat', 'muhat0', 'mu0'.
    
    Returns:
        out: the loss coefficient for the given row
    """
    out = (((data[A] == a) & (data[R] == r)) * (1 - 2 * data[outcome])).mean()

    return out


def risk_coefs(data, A='A', R='R', outcome='phihat'):
    """Compute the risk coefficients $\beta_{a, r}$ for all a, r.

    See `risk_coef` for more details.

    Args:
        data: DataFrame containing columns A, R, and the outcome. In general,
            this will be the output of train_nuisance, and the outcome will be
            'phihat'.
        A: name of the sensitive feature column (str).
        R: name of the risk prediction column (str).
        outcome: name of the outcome column, one of 'phihat', 'muhat0', 'mu0'.
    
    Returns:
        out: a 4-element array containing the coefficients for the loss function.
    """
    ar_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    coefs = [risk_coef(data, a, r, A, R, outcome) for a, r in ar_list]
    out = np.array(coefs).clip(-1, 1)

    return out


#################################################################
#### Error rate functions, for the optimization constraints #####
#################################################################

def est_cFPR_overall(data, R, outcome='phihat'):
    """Compute the estimated counterfactual False Positive Rate.
    
    The outcome should be one of 'phihat', 'muhat0', 'mu0'. In general, `data` 
    will be generated by `train_nuisance`, in which case 'phihat' will
    represent doubly robust pseudo-outcomes under the baseline treatment level,
    while muhat0 will represent the outcome regression under the baseline
    treatment level. In this case, using phihat will yield a doubly robust
    estimator of the risk, while using muhat0 will yield a plugin estimator of
    the risk. In general, doubly robust estimators are preferred unless the
    outcome regression model is known to be correctly specified.

    Args:
        data: DataFrame containing columns A, R, and the outcome. In general,
            this will be the output of train_nuisance, and the outcome will be
            'phihat'.
        A: name of the sensitive feature column (str).
        R: name of the risk prediction column (str).
        outcome: name of the outcome column, one of 'phihat', 'muhat0', 'mu0'.
    
    Returns:
        out: the estimated counterfactual False Positive rate.
    """
    out = data.eval(f"{R}*(1 - {outcome})").mean() / data.eval(f"1 - {outcome}").mean()
    return out


def est_cFNR_overall(data, R, outcome='phihat'):
    """Compute estimated counterfactual False Positive Rate.
    
    The outcome should be one of 'phihat', 'muhat0', 'mu0'. In general, `data` 
    will be generated by `train_nuisance`, in which case 'phihat' will
    represent doubly robust pseudo-outcomes under the baseline treatment level,
    while muhat0 will represent the outcome regression under the baseline
    treatment level. In this case, using phihat will yield a doubly robust
    estimator of the risk, while using muhat0 will yield a plugin estimator of
    the risk. In general, doubly robust estimators are preferred unless the
    outcome regression model is known to be correctly specified.

    Args:
        data: DataFrame containing columns A, R, and the outcome. In general,
            this will be the output of train_nuisance, and the outcome will be
            'phihat'.
        A: name of the sensitive feature column (str).
        R: name of the risk prediction column (str).
        outcome: name of the outcome column, one of 'phihat', 'muhat0', 'mu0'.
    
    Returns:
        out: the estimated counterfactual False Negative rate.
    """
    out = data.eval(f"(1 - {R})*{outcome}").mean() / data.eval(f"{outcome}").mean()
    return out


def fairness_coefs(data, A='A', R='R', outcome='phihat'):
    """Get coefficients that define the fairness constraints for the linear program.

    The outcome should be one of 'phihat', 'muhat0', 'mu0'. In general, `data` 
    will be generated by `train_nuisance`, in which case 'phihat' will
    represent doubly robust pseudo-outcomes under the baseline treatment level,
    while muhat0 will represent the outcome regression under the baseline
    treatment level. In this case, using phihat will yield a doubly robust
    estimator of the risk, while using muhat0 will yield a plugin estimator of
    the risk. In general, doubly robust estimators are preferred unless the
    outcome regression model is known to be correctly specified.

    Args:
        data: DataFrame containing columns A, R, and the outcome. In general,
            this will be the output of train_nuisance, and the outcome will be
            'phihat'.
        A: name of the sensitive feature column (str).
        R: name of the risk prediction column (str).
        outcome: name of the outcome column, one of 'phihat', 'muhat0', 'mu0'.
    
    Returns:
        out: a tuple containing the coefficients for the constraints related to
            the counterfactual false positive and false negative rates.
    """
    false_pos = data.groupby(A).apply(est_cFPR_overall, R, outcome).values.clip(0, 1)
    false_neg = data.groupby(A).apply(est_cFNR_overall, R, outcome).values.clip(0, 1)
    coefs_pos = np.array(
        [1 - false_pos[0], false_pos[0], false_pos[1] - 1, -false_pos[1]])
    coefs_neg = np.array(
        [-false_neg[0], false_neg[0] - 1, false_neg[1], 1 - false_neg[1]])

    return (coefs_pos, coefs_neg)


#######################
#### Optimization #####
#######################

def optimize(risk_coefs_, coefs_pos, coefs_neg, epsilon_pos, epsilon_neg):
    """Solve the Linear Program.
    
    Args:
        risk_coefs: coefficients for the loss function
        coefs_pos: coefficients for the fairness constraints on the false
            positive rate.
        coefs_neg: coefficients for the fairness constraints on the false
            negative rate.
        epsilon_pos: maximum allowed violation of the false positive rate
        epsilon_neg: maximum allowed violation of the false negative rate
    """
    theta = cp.Variable(4)
    objective = cp.Minimize(risk_coefs_ @ theta)
    constraints = [0 <= theta, theta <= 1,
                   coefs_pos @ theta <= epsilon_pos,
                   coefs_pos @ theta >= -epsilon_pos,
                   coefs_neg @ theta <= epsilon_neg,
                   coefs_neg @ theta >= -epsilon_neg]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS_BB)

    return theta.value


def fair_derived(data, A, X, R, D, Y, learner_pi, learner_mu, epsilon_pos,
                 epsilon_neg, outcome='phihat', test_size=0.50, trunc_pi=0.975):
    """Optimization with chosen estimators for loss and fairness constraints.
    
    Args:
        data: DataFrame containing columns A, X, R, D, Y.
        A: name of the sensitive feature column (str).
        X: names of the feature columns (list).
        R: name of the risk score column (str).
        D: name of the treatment column, aka the decision column (str).
        Y: name of the outcome column (str).
        learner_pi: learner for the propensity score (sklearn model)
        learner_mu: learner for the outcome model (sklearn model)
        epsilon_pos: maximum allowed violation of the false positive rate
        epsilon_neg: maximum allowed violation of the false negative rate
        outcome: name of the outcome column, one of 'phihat', 'muhat0', 'mu0'.
            This comes from the nuisance parameter estimation.
        test_size: proportion of the data to use as a test set
        trunc_pi: truncation level for the propensity score, a number in (0, 1), 
            default 0.975. Propensity scores are clipped to be at most this value.
            This is useful for preventing numerical instability.

    Returns:
        out: a dictionary containing the optimization results, including the
            optimal decision variable theta, the risk coefficients, and the
            fairness coefficients.
    """
    data_nuisance, data_opt = train_test_split(data, test_size=test_size)
    nuis = train_nuisance(data_nuisance, data_opt, A, X, R, D, Y, learner_pi,
                          learner_mu, trunc_pi)
    data_opt = pd.concat([data_opt.reset_index(drop=True), nuis.reset_index(drop=True)], axis=1)
    obj = risk_coefs(data_opt, A=A, R=R, outcome=outcome)
    fair_pos, fair_neg = fairness_coefs(data_opt, A=A, R=R, outcome=outcome)
    theta = optimize(obj, fair_pos, fair_neg, epsilon_pos, epsilon_neg)

    out = {'theta': theta, 'risk_coefs': obj,
           'fairness_coefs_pos': fair_pos, 'fairness_coefs_neg': fair_neg}

    return out
