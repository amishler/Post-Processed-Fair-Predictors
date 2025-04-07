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
    """Compute a confidence interval around a probability estimate, 
    either on the expit (probability) or the logit (log-odds) scale.

    Args:
        est (float): Point estimate in (0, 1).
        sd (float): Standard deviation of the estimate (on probability scale).
        z (float): Z-score for desired confidence level (e.g., 1.96 for 95% CI).
        n (int): Sample size (used in standard error calculation).
        scale (str): 'logit' or 'expit'.

    Returns:
        (float, float): Lower and upper confidence bounds (on probability scale).
    """
    se = sd / np.sqrt(n)

    if scale == 'logit' and 0 < est < 1:
        # Apply delta method to get SE on logit scale
        se_logit = se / (est * (1 - est))
        logit_est = logit(est)
        lower = expit(logit_est - z * se_logit)
        upper = expit(logit_est + z * se_logit)
    else:
        # Use normal CI on probability scale and clip to [0,1]
        lower = max(0, est - z * se)
        upper = min(1, est + z * se)

    return lower, upper


def ci_prob_diff(est, sd, z, n, scale='logit'):
    """
    Compute a confidence interval for the difference between two probabilities.

    Args:
        est (float): Difference estimate in [-1, 1].
        sd (float): Standard deviation of the difference (on prob scale).
        z (float): Z-score for CI.
        n (int): Sample size.
        scale (str): 'logit' (transform scale) or 'expit' (probability scale).

    Returns:
        (float, float): CI bounds for the probability difference.
    """
    se = sd / np.sqrt(n)

    if scale == 'logit' and -1 < est < 1:
        # Map est ∈ [-1, 1] to pseudo-probability q ∈ [0, 1]
        q = est / 2 + 0.5

        # Delta method: derivative of logit(q) w.r.t. est is 1 / [2 * q * (1 - q)]
        dq_dest = 0.5
        se_logit = se * dq_dest / (q * (1 - q))

        # CI on logit scale, back-transform
        lower_q = expit(logit(q) - z * se_logit)
        upper_q = expit(logit(q) + z * se_logit)

        # Map back to [-1, 1]
        lower = 2 * (lower_q - 0.5)
        upper = 2 * (upper_q - 0.5)
    else:
        lower = max(-1, est - z * se)
        upper = min(1, est + z * se)

    return lower, upper


def est_risk_post(theta, data, A='A', R='R', outcome='phihat', ci=0.95,
             ci_scale='logit'):
    """Estimate overall risk and the change in risk due to post-processing.

    Args:
        theta (np.ndarray): 4-element vector indexing the post-processed
            predictor.
        data (pd.DataFrame): Evaluation data.
        A, R (str): Column names for sensitive attribute and risk score.
        outcome (str): Column with true outcome Y0 or pseudo-outcome (phihat, 
            muhat0, or mu0).
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


def est_cFPR_post(theta, data, A='A', R='R', outcome='phihat', ci=0.95,
             ci_scale='logit'):
    """
    Estimate counterfactual FPRs and their gap for the post-processed aka
    derived predictor.

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
    # Compute all three estimates, for FPR0, FPR1, and their gap.
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


def est_cFNR_post(theta, data, A='A', R='R', outcome='phihat', ci=0.95,
             ci_scale='logit'):
    """
    Estimate counterfactual FNRs and their gap for the post-processed aka
    derived predictor.


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


def est_predictive_change(theta, data, A='A', R='R', ci=0.95):
    """
    Estimate the predictive change due to post-processing, defined as the 
    probability that the derived predictions differ from the original 
    predictions.

    Args:
        theta (np.ndarray): 4-element vector indexing the post-processed
            predictor.
        data (pd.DataFrame): Evaluation data.
        A, R, outcome (str): Column names for the sensitive feature and the
            input predictor.
        ci (float): Confidence level, or None in which case no CI will be
            computed.
    
    Returns:
        pd.DataFrame: A single row DataFrame with the metric 'prop_differ', its 
            point estimate, and optional confidence intervals.
    """
    ind_df = indicator_df(data, A, R).astype(int)
    newvar = ind_df.dot([theta[0], 1 - theta[1], theta[2], 1 - theta[3]])
    diff_est = newvar.mean()       
    out = pd.DataFrame({'metric': 'prop_differ', 'value': diff_est, 
                       'ci_lower': None, 'ci_upper': None}, index = [0])
    if ci:    
        n = data.shape[0]
        z = scipy.stats.norm.ppf((ci + 1) / 2)
        diff_sd = np.std(newvar)
        ci_lower = diff_est - z*diff_sd/np.sqrt(n)
        ci_upper = diff_est + z*diff_sd/np.sqrt(n)
        out['ci_lower'] = np.max([ci_lower, 0])
        out['ci_upper'] = np.min([ci_upper, 1])
    
    return out


def metrics_post(theta, data, A='A', R='R', outcome='phihat', ci=0.95, ci_scale='logit'):
    """
    Compute risk, FPR, and FNR metrics for a given post-processed aka derived
    predictor.

    Args:
        theta (np.ndarray): 4-element vector indexing the post-processed
            predictor.
        data (pd.DataFrame): Evaluation dataset.
        A, R (str): Sensitive attribute and risk score column names.
        outcome (str): Column with estimated or true outcome. Could be 'Y0', 
          'phihat', 'muhat0', or 'mu0' for example.
        ci (float): Confidence level for intervals.
        ci_scale (str): CI transformation scale.

    Returns:
        pd.DataFrame: Combined metric estimates and confidence intervals.
    """
    risk = est_risk_post(theta, data, A=A, R=R, outcome=outcome, ci=ci, ci_scale=ci_scale)
    cFPR = est_cFPR_post(theta, data, A=A, R=R, outcome=outcome, ci=ci, ci_scale=ci_scale)
    cFNR = est_cFNR_post(theta, data, A=A, R=R, outcome=outcome, ci=ci, ci_scale=ci_scale)
    out = pd.concat([risk, cFPR, cFNR])

    return out


def coverage(metrics_est, metrics_true, simplify=True):
    """
    Calculate CI coverage rate based on true metric values.

    Args:
        metrics_est (pd.DataFrame): DataFrame with estimated CIs.
        metrics_true (pd.DataFrame): DataFrame with true metric values.
        simplify (bool): If True, return group-averaged coverage by metric and n.
                        Requires 'n' to be a column in `metrics_est`.

    Returns:
        pd.DataFrame: Either row-wise coverage or grouped summary.

    Raises:
        ValueError: If simplify=True but 'n' is not a column in metrics_est.
    """
    # Convert {metric: [value]} → {metric: value}
    true_dict = (
        metrics_true[['metric', 'value']]
        .set_index('metric')['value']
        .to_dict()
    )

    # Map metric → scalar value
    true_vals = metrics_est['metric'].map(true_dict)

    # Compute coverage
    cov = (metrics_est['ci_lower'] <= true_vals) & (true_vals <= metrics_est['ci_upper'])

    # Attach coverage column
    out = metrics_est.assign(coverage=cov)

    if simplify:
        if 'n' not in metrics_est.columns:
            raise ValueError("Cannot simplify coverage summary: column 'n' not found in `metrics_est`.")
        out = out.groupby(['metric', 'n'])[['coverage']].mean().reset_index()

    return out
