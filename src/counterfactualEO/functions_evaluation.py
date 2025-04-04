"""
functions_evaluation.py

This module provides tools to evaluate post-processed fair predictors.
This includes estimation of risk and fairness metrics with influence 
function-based asymptotically valid confidence intervals.
"""

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit

from counterfactualEO.functions_estimation import fairness_coefs


def indicator_df(df, A='A', R='R'):
    """
    Return a one-hot encoded DataFrame for each (A, R) pair.

    Args:
        df (pd.DataFrame): Dataset containing binary columns A and R.
        A (str): Column name for sensitive attribute.
        R (str): Column name for risk score.

    Returns:
        pd.DataFrame: Four-column indicator DataFrame: 
            ['A0_R0', 'A0_R1', 'A1_R0', 'A1_R1'].
    """
    ar_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    out_list = [((df[A] == a) & (df[R] == r)) for a, r in ar_list]
    out = pd.DataFrame(np.array(out_list).T,
                       columns = ['A0_R0', 'A0_R1', 'A1_R0', 'A1_R1'])
    return out


def ci_prob(est, sd, z, n, scale='logit'):
    """Compute a CI around a probability estimate, either on the expit or the 
    logit scale.

    Args:
        est (float): Point estimate in [0, 1].
        sd (float): Standard deviation of the estimate.
        z (float): Z-score for desired confidence level.
        n (int): Sample size.
        scale (str): Either 'logit' (for CI in logit space) or 'expit' 
            (for CI in [0, 1]).

    Returns:
        (float, float): Lower and upper confidence bounds.
    """
    if (scale == 'logit') and (0 < est < 1):
        sd = sd/(est*(1 - est))
        lower = expit(logit(est) - z*sd/np.sqrt(n))
        upper = expit(logit(est) + z*sd/np.sqrt(n))
    else:
        lower = max(0, est - z * sd / np.sqrt(n))
        upper = min(est + z * sd / np.sqrt(n), 1)

    return lower, upper


def ci_prob_diff(est, sd, z, n, scale='logit'):
    """
    Compute a confidence interval for the difference between two probabilities.

    Args:
        est (float): Difference estimate in [-1, 1].
        sd (float): Standard deviation of the difference.
        z (float): Z-score for CI.
        n (int): Sample size.
        scale (str): Either 'logit' (for CI in logit space) or 'expit' 
            (for CI in [0, 1]).

    Returns:
        (float, float): CI bounds for the probability difference.
    """
    if (scale == 'logit') and (-1 < est < 1):
        sd = sd*2/(1 - est**2)
        lower = 2*(expit(logit(est/2 + 1/2) - z * sd / np.sqrt(n)) - 1/2)
        upper = 2*(expit(logit(est/2 + 1/2) + z * sd / np.sqrt(n)) - 1/2)
    else:
        lower = max(-1, est - z * sd / np.sqrt(n))
        upper = min(est + z * sd / np.sqrt(n), 1)

    return lower, upper


def est_risk(theta, data, A='A', R='R', outcome='phihat', ci=0.95,
             ci_scale='logit'):
    """Estimate overall risk and the change in risk due to post-processing.

    Args:
        theta (np.ndarray): 4-element vector indexing the post-processed
            predictor.
        data (pd.DataFrame): Evaluation data.
        A, R (str): Column names for sensitive attribute and risk score.
        outcome (str): Column with pseudo-outcome (phihat, muhat0, or mu0).
        ci (float): Confidence level (e.g. 0.95), or None for no CI.
        ci_scale (str): CI transformation scale ('logit' or 'expit').

    Returns:
        pd.DataFrame: Rows for risk and risk_change with CIs.
    """
    ind_df = indicator_df(data, A, R)

    ## Risk and risk change point estimates, using influence function
    inf_risk_pre = ind_df.dot([0, 1, 0, 1]) * (1 - 2 * data[outcome]) + data[
        outcome]
    inf_risk_post = ind_df.dot(theta) * (1 - 2 * data[outcome]) + data[outcome]
    inf_change = inf_risk_post - inf_risk_pre

    risk_est = min(max(0, inf_risk_post.mean()), 1)
    change_est = min(max(-1, inf_change.mean()), 1)
    out = {'metric': ['risk', 'risk_change'], 'value': [risk_est, change_est],
           'ci_lower': [None] * 2, 'ci_upper': [None] * 2}

    ## CIs
    if ci:
        z = scipy.stats.norm.ppf((ci + 1) / 2)
        n = data.shape[0]
        risk_sd = np.std(inf_risk_post)
        change_sd = np.std(inf_change)
        lower_risk, upper_risk = ci_prob(risk_est, risk_sd, z, n, ci_scale)
        lower_change, upper_change = ci_prob_diff(change_est, change_sd, z, n, ci_scale)

        out['ci_lower'] = [lower_risk, lower_change]
        out['ci_upper'] = [upper_risk, upper_change]

    return pd.DataFrame(out)


def est_cFPR_group(theta, data, A='A', R='R', outcome='phihat', ci=0.95,
             ci_scale='logit'):
    """
    Estimate counterfactual FPRs for groups A=0 and A=1, and their gap.

    Args:
        theta (np.ndarray): 4-element vector indexing the post-processed
            predictor.
        data (pd.DataFrame): Evaluation data.
        A, R, outcome (str): Column names.
        ci (float): Confidence level (optional).
        ci_scale (str): 'logit' or 'expit'.

    Returns:
        pd.DataFrame: FPR0, FPR1, and gap_FPR with optional CIs.
    """
    # Get the two vectors of influence function values for the estimators.
    # Compute all three estimates.
    # If CI, get the variance vectors for both groups. Compute all 3 variances.
    coefs_pos, _ = fairness_coefs(data, A, R, outcome)
    est0 = coefs_pos[:2] @ theta[:2]
    est1 = abs(coefs_pos[2:] @ theta[2:])
    est_diff = est0 - est1

    out = {'metric': ['FPR0', 'FPR1', 'gap_FPR'],
           'value': [est0, est1, est_diff]}
    out['ci_lower'] = [None] * 3
    out['ci_upper'] = [None] * 3

    if ci:
        n = data.shape[0]
        z = scipy.stats.norm.ppf((ci + 1) / 2)
        h0 = (1 - data[outcome]) * (1 - data[A])
        h1 = (1 - data[outcome]) * data[A]
        var_func0 = 1 / h0.mean() * (theta[1] - theta[0]) * (
                    data[R] - est0) * h0
        var_func1 = 1 / h1.mean() * (theta[3] - theta[2]) * (
                    data[R] - est0) * h1
        sd0 = np.std(var_func0)
        sd1 = np.std(var_func1)
        sd_diff = np.std(var_func0 - var_func1)

        lower0, upper0 = ci_prob(est0, sd0, z, n, ci_scale)
        lower1, upper1 = ci_prob(est1, sd1, z, n, ci_scale)
        lower_diff, upper_diff = ci_prob_diff(est_diff, sd_diff, z, n, ci_scale)

        out['ci_lower'] = [lower0, lower1, lower_diff]
        out['ci_upper'] = [upper0, upper1, upper_diff]

    return pd.DataFrame(out)


def est_cFNR_group(theta, data, A='A', R='R', outcome='phihat', ci=0.95,
             ci_scale='logit'):
    """
    Estimate counterfactual FNRs for groups A=0 and A=1, and their gap.

    Args:
        theta (np.ndarray): 4-element vector indexing the post-processed
            predictor.
        data (pd.DataFrame): Evaluation data.
        A, R, outcome (str): Column names.
        ci (float): Confidence level.
        ci_scale (str): CI scale ('logit' or 'expit').

    Returns:
        pd.DataFrame: FNR0, FNR1, and gap_FNR with optional CIs.
    """
    # Get the two vectors of influence function values for the estimators.
    # Compute all three estimates.
    # If CI, get the variance vectors for both groups. Compute all 3 variances.
    _, coefs_neg = fairness_coefs(data, A, R, outcome)
    est0 = coefs_neg[:2] @ theta[:2] + 1
    est1 = 1 - coefs_neg[2:] @ theta[2:]
    est_diff = est0 - est1

    out = {'metric': ['FNR0', 'FNR1', 'gap_FNR'],
           'value': [est0, est1, est_diff]}
    out['ci_lower'] = [None] * 3
    out['ci_upper'] = [None] * 3

    if ci:
        n = data.shape[0]
        z = scipy.stats.norm.ppf((ci + 1) / 2)
        h0 = data[outcome] * (1 - data[A])
        h1 = data[outcome] * data[A]
        var_func0 = 1 / h0.mean() * (theta[1] - theta[0]) * (
                    data[R] - est0) * h0
        var_func1 = 1 / h1.mean() * (theta[3] - theta[2]) * (
                    data[R] - est0) * h1
        sd0 = np.std(var_func0)
        sd1 = np.std(var_func1)
        sd_diff = np.std(var_func0 - var_func1)

        lower0, upper0 = ci_prob(est0, sd0, z, n, ci_scale)
        lower1, upper1 = ci_prob(est1, sd1, z, n, ci_scale)
        lower_diff, upper_diff = ci_prob_diff(est_diff, sd_diff, z, n, ci_scale)

        out['ci_lower'] = [lower0, lower1, lower_diff]
        out['ci_upper'] = [upper0, upper1, upper_diff]

    return pd.DataFrame(out)


def metrics(theta, data, A='A', R='R', outcome='phihat', ci=0.95, ci_scale='logit'):
    """
    Compute risk, FPR, and FNR metrics for a given predictor.

    Args:
        theta (np.ndarray): 4-element vector indexing the post-processed
            predictor.
        data (pd.DataFrame): Evaluation dataset.
        A, R (str): Sensitive attribute and risk score column names.
        outcome (str): Column with estimated or true outcome.
        ci (float): Confidence level for intervals.
        ci_scale (str): CI transformation scale.

    Returns:
        pd.DataFrame: Combined metric estimates and confidence intervals.
    """
    risk = est_risk(theta, data, A=A, R=R, outcome=outcome, ci=ci, ci_scale=ci_scale)
    cFPR = est_cFPR_group(theta, data, A=A, R=R, outcome=outcome, ci=ci, ci_scale=ci_scale)
    cFNR = est_cFNR_group(theta, data, A=A, R=R, outcome=outcome, ci=ci, ci_scale=ci_scale)
    out = pd.concat([risk, cFPR, cFNR])

    return out


def _metrics_to_df(res):
    """
    Convert 3D array of simulation results into a long-format DataFrame.

    Args:
        res (np.ndarray): Array of shape (mc_reps, num_metrics, 5) for one n.

    Returns:
        pd.DataFrame: Flattened result with 'mc_iter' and metric columns.
    """
    out = [pd.DataFrame(res[i, :, :]) for i in range(res.shape[0])]
    out = pd.concat(out, keys=list(range(len(out))), names=['mc_iter'])
    out = out.reset_index().drop(columns='level_1')
    out.columns = ['mc_iter', 'metric', 'value', 'ci_lower', 'ci_upper']

    return out


def metrics_to_df(res, n_arr, setting, data_val, ci=0.95):
    """
    Convert simulation output across sample sizes into one plottable DataFrame.

    Args:
        res (list): List of results from multiple `sim_theta` runs.
        n_arr (list): Sample sizes corresponding to each result.
        setting (str): Label for experimental setting.
        data_val (pd.DataFrame): Data used to re-evaluate each theta.
        ci (float): Confidence level.

    Returns:
        pd.DataFrame: Long-format table with metrics, CIs, and metadata.
    """
    out = [np.apply_along_axis(metrics, 1, rr['theta_arr'], data=data_val,
                               outcome='mu0', ci=ci) for rr in res]
    out = pd.concat([_metrics_to_df(arr) for arr in out], keys=n_arr)
    out = out.reset_index().drop(columns='level_1')
    out.columns = ['n', 'mc_iter', 'metric', 'value', 'ci_lower', 'ci_upper']
    out['setting'] = setting
    out['value'] = pd.to_numeric(out['value'])

    return out


def coverage(metrics_est, metrics_true, simplify=True):
    """
    Calculate CI coverage rate based on true metric values.

    Args:
        metrics_est (pd.DataFrame): DataFrame with estimated CIs.
        metrics_true (pd.DataFrame): DataFrame with true metric values.
        simplify (bool): If True, return group-averaged coverage by metric and n.

    Returns:
        pd.DataFrame: Either row-wise coverage or grouped summary.
    """
    true_dict = metrics_true[['metric', 'value']].set_index('metric').T.to_dict(
        'list')
    true_vals = metrics_est['metric'].replace(true_dict)
    cov = (metrics_est['ci_lower'] <= true_vals) & (
                true_vals <= metrics_est['ci_upper'])
    out = metrics_est.assign(coverage=cov)
    if simplify:
        out = out.groupby(['metric', 'n'])[['coverage']].mean()
        out = out.unstack().reset_index().rename(columns = {'': 'metric'})
        out.columns = out.columns.droplevel(0)

    return out
