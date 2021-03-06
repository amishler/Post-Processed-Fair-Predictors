import cvxpy as cp
import itertools
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from sklearn.model_selection import train_test_split


def train_nuisance(train, test, A, X, R, D, Y, learner_pi, learner_mu,
                   trunc_pi=0.975):
    """Train nuisance regressions."""
    train = train.reset_index()
    test = test.reset_index()
    pred_cols = list(itertools.chain.from_iterable([A, X, R]))
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


#########################
#### Risk functions #####
#########################

def risk_coef(data, a, r, A='A', R='R', outcome='phihat'):
    """Compute the loss coefficient for a single row, for given values
    A = a, R = r.

    Using phihat as the outcome yields a doubly robust estimator.
    Using muhat0 as the outcome yields a plugin estimator.
    Using mu0 as the outcome yields (something close to) the ``true'' loss
    coefficient.
    """
    out = (((data[A] == a) & (data[R] == r)) * (1 - 2 * data[outcome])).mean()

    return out


def risk_coefs(data, A='A', R='R', outcome='phihat'):
    """Compute the loss coefficients.

    Using phihat as the outcome yields a doubly robust estimator.
    Using muhat0 as the outcome yields a plugin estimator.
    Using mu0 as the outcome yields (something close to) the ``true'' loss
    coefficients.
    """
    ar_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    coefs = [risk_coef(data, a, r, A, R, outcome) for a, r in ar_list]
    out = np.array(coefs).clip(-1, 1)

    return out


###############################
#### Error rate functions #####
###############################

# def cFPR(data, R, outcome='phihat'):
#     """Compute estimated counterfactual False Positive Rate."""
#     return (data[R]*(1 - data[outcome])).mean() / (1 - data[outcome]).mean()

# def cFPR(data, A, R, outcome):
#     """Compute estimated group-specific counterfactual False Positive Rate."""
#     return data.groupby(A).apply(_cFPR, R, outcome).values

# def cFNR(data, R, outcome='phihat'):
#     """Compute estimated counterfactual False Positive Rate."""
#     return ((1 - data[R])*data[outcome]).mean() / data[outcome].mean()

# def cFNR(data, A, R, outcome):
#     """Compute estimated group-specific counterfactual False Positive Rate."""
#     return data.groupby(A).apply(_cFNR, R, outcome).values

def cFPR(data, R, outcome='phihat'):
    """Compute estimated counterfactual False Positive Rate."""
    out = data.eval("{}*(1 - {})".format(R, outcome)).mean() / data.eval("1 - {}".format(outcome)).mean()
    return out

def cFNR(data, R, outcome='phihat'):
    """Compute estimated counterfactual False Positive Rate."""
    out = data.eval("(1 - {})*{}".format(R, outcome)).mean() / data.eval("{}".format(outcome)).mean()
    return out

def fairness_coefs(data, A='A', R='R', outcome='phihat'):
    """Get coefficients that define the fairness constraints for the estimator."""
    false_pos = data.groupby(A).apply(cFPR, R, outcome).values.clip(0, 1)
    false_neg = data.groupby(A).apply(cFNR, R, outcome).values.clip(0, 1)
    coefs_pos = np.array(
        [1 - false_pos[0], false_pos[0], false_pos[1] - 1, -false_pos[1]])
    coefs_neg = np.array(
        [-false_neg[0], false_neg[0] - 1, false_neg[1], 1 - false_neg[1]])

    return (coefs_pos, coefs_neg)


def optimize(risk_coefs, coefs_pos, coefs_neg, epsilon_pos, epsilon_neg):
    """Solve the LP."""
    theta = cp.Variable(4)
    objective = cp.Minimize(risk_coefs @ theta)
    constraints = [0 <= theta, theta <= 1,
                   coefs_pos @ theta <= epsilon_pos,
                   coefs_pos @ theta >= -epsilon_pos,
                   coefs_neg @ theta <= epsilon_neg,
                   coefs_neg @ theta >= -epsilon_neg]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.ECOS_BB)

    return theta.value  # , result


#### Full optimization ####
def fair_derived(data, A, X, R, D, Y, learner_pi, learner_mu, epsilon_pos,
                 epsilon_neg, outcome='phihat', test_size=0.50, trunc_pi=0.975):
    """Optimization with chosen estimators for loss and fairness constraints."""
    data_nuisance, data_opt = train_test_split(data, test_size=test_size)
    nuis = train_nuisance(data_nuisance, data_opt, A, X, R, D, Y, learner_pi,
                          learner_mu, trunc_pi)
    data_opt = pd.concat([data_opt.reset_index(), nuis.reset_index()], axis=1)
    obj = risk_coefs(data_opt, A=A, R=R, outcome=outcome)
    fair_pos, fair_neg = fairness_coefs(data_opt, A=A, R=R, outcome=outcome)
    theta = optimize(obj, fair_pos, fair_neg, epsilon_pos, epsilon_neg)

    out = {'theta': theta, 'risk_coefs': obj,
           'fairness_coefs_pos': fair_pos, 'fairness_coefs_neg': fair_neg}

    return out


def indicator_df(df, A='A', R='R'):
    """Get four-column df indicating values of A and R in each row."""
    ar_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    out_list = [((df[A] == a) & (df[R] == r)) for a, r in ar_list]
    out = pd.DataFrame(np.array(out_list).T,
                       columns = ['A0_R0', 'A0_R1', 'A1_R0', 'A1_R1'])
    return out


def ci_prob(est, sd, z, n, scale='logit'):
    """Compute a CI around a probability, either on the expit or the logit scale."""
    if (scale == 'logit') and (0 < est < 1):
        sd = sd/(est*(1 - est))
        lower = expit(logit(est) - z*sd/np.sqrt(n))
        upper = expit(logit(est) + z*sd/np.sqrt(n))
    else:
        lower = max(0, est - z * sd / np.sqrt(n))
        upper = min(est + z * sd / np.sqrt(n), 1)

    return lower, upper


def ci_prob_diff(est, sd, z, n, scale='logit'):
    """Compute a CI around a difference of probabilities, either on the expit or the logit scale."""
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
    """Compute risk and risk change estimates, with optional CIs."""
    ind_df = indicator_df(data, A, R)

    ## Risk and risk change point estimates
    # influence function
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

        # if (ci_scale == 'logit') and (0 < risk_est < 1):
        #     risk_sd = np.std(inf_risk_post)/(risk_est*(1 - risk_est))
        #     lower_risk = expit(logit(risk_est) - z * risk_sd / np.sqrt(n))
        #     upper_risk = expit(logit(risk_est) + z * risk_sd / np.sqrt(n))
        # else:
        #     risk_sd = np.std(inf_risk_post)
        #     lower_risk = max(0, risk_est - z * risk_sd / np.sqrt(n))
        #     upper_risk = min(risk_est + z * risk_sd / np.sqrt(n), 1)
        # if (ci_scale == 'logit') and (-1 < change_est < 1):
        #     change_sd = np.std(inf_change)*2/(1 - change_est**2)
        #     lower_change = 2*(expit(logit(change_est/2 + 1/2) - z * change_sd / np.sqrt(n))) - 1
        #     upper_change = 2*(expit(logit(change_est/2 + 1/2) + z * change_sd / np.sqrt(n))) - 1
        # else:
        #     change_sd = np.std(inf_change)
        #     lower_change = max(-1, change_est - z * change_sd / np.sqrt(n))
        #     upper_change = min(change_est + z * change_sd / np.sqrt(n), 1)

        out['ci_lower'] = [lower_risk, lower_change]
        out['ci_upper'] = [upper_risk, upper_change]

    return pd.DataFrame(out)


# def est_risk(theta, data, A='A', R='R', outcome='phihat', ci=0.95):
#     """Compute risk and risk change estimates, with optional CIs."""
#     ind_df = indicator_df(data, A, R)
#
#     ## Risk and risk change point estimates
#     # influence function
#     inf_risk_pre = ind_df.dot([0, 1, 0, 1]) * (1 - 2 * data[outcome]) + data[
#         outcome]
#     inf_risk_post = ind_df.dot(theta) * (1 - 2 * data[outcome]) + data[outcome]
#     inf_change = inf_risk_post - inf_risk_pre
#
#     risk_est = min(max(0, inf_risk_post.mean()), 1)
#     change_est = min(max(-1, inf_change.mean()), 1)
#     out = {'metric': ['risk', 'risk_change'], 'value': [risk_est, change_est],
#            'ci_lower': [None] * 2, 'ci_upper': [None] * 2}
#
#     ## CIs
#     if ci:
#         z = scipy.stats.norm.ppf((ci + 1) / 2)
#         n = data.shape[0]
#         risk_sd = np.std(inf_risk_post)
#         change_sd = np.std(inf_change)
#
#         lower_risk = max(0, risk_est - z * risk_sd / np.sqrt(n))
#         upper_risk = min(risk_est + z * risk_sd / np.sqrt(n), 1)
#         lower_change = max(-1, change_est - z * change_sd / np.sqrt(n))
#         upper_change = min(change_est + z * change_sd / np.sqrt(n), 1)
#
#         out['ci_lower'] = [lower_risk, lower_change]
#         out['ci_upper'] = [upper_risk, upper_change]
#
#     return pd.DataFrame(out)


def est_cFPR(theta, data, A='A', R='R', outcome='phihat', ci=0.95,
             ci_scale='logit'):
    """Estimate the cFPR and the fairness gap for one group.

    Need to be using theta, which I'm not currently.
    """
    # Get the two vectors of IF values for the estimators.
    # Compute all three estimates
    # If CI, get the variance vectors for both groups. Compute all 3 variances.
    ## cFPRs for the input predictor
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
        # lower0 = max(0, est0 - z * sd0 / np.sqrt(n))
        # upper0 = min(est0 + z * sd0 / np.sqrt(n), 1)
        # lower1 = max(0, est1 - z * sd1 / np.sqrt(n))
        # upper1 = min(est1 + z * sd1 / np.sqrt(n), 1)
        # lower_diff = max(-1, est_diff - z * sd_diff / np.sqrt(n))
        # upper_diff = min(est_diff + z * sd_diff / np.sqrt(n), 1)

        out['ci_lower'] = [lower0, lower1, lower_diff]
        out['ci_upper'] = [upper0, upper1, upper_diff]

    return pd.DataFrame(out)


def est_cFNR(theta, data, A='A', R='R', outcome='phihat', ci=0.95,
             ci_scale='logit'):
    """Estimate the cFNR and the fairness gap for one group.

    Need to be using theta, which I'm not currently.
    """
    # Get the two vectors of IF values for the estimators.
    # Compute all three estimates
    # If CI, get the variance vectors for both groups. Compute all 3 variances.
    ## cFPRs for the input predictor
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
        var_func_diff = var_func0 - var_func1
        sd0 = np.std(var_func0)
        sd1 = np.std(var_func1)
        sd_diff = np.std(var_func0 - var_func1)

        lower0, upper0 = ci_prob(est0, sd0, z, n, ci_scale)
        lower1, upper1 = ci_prob(est1, sd1, z, n, ci_scale)
        lower_diff, upper_diff = ci_prob_diff(est_diff, sd_diff, z, n, ci_scale)

        # lower0 = max(0, est0 - z * sd0 / np.sqrt(n))
        # upper0 = min(est0 + z * sd0 / np.sqrt(n), 1)
        # lower1 = max(0, est1 - z * sd1 / np.sqrt(n))
        # upper1 = min(est1 + z * sd1 / np.sqrt(n), 1)
        # lower_diff = max(-1, est_diff - z * sd_diff / np.sqrt(n))
        # upper_diff = min(est_diff + z * sd_diff / np.sqrt(n), 1)

        out['ci_lower'] = [lower0, lower1, lower_diff]
        out['ci_upper'] = [upper0, upper1, upper_diff]

    return pd.DataFrame(out)


# def _est_cFPR(theta, data, A='A', R='R', outcome='phihat', ci='None'):
#     """Estimate the cFPR and the fairness gap for one group."""
#     inf_func = (data[R]*(1 - data[outcome])).mean() / (1 - data[outcome])
#     est = inf_func.mean()
#     out = {'est': est, 'ci_lower': None, 'ci_upper': None}
#     if ci:
#         z = scipy.stats.norm.ppf((ci + 1)/2)
#         sd = np.sd(inf_func)
#         n = data.shape[0]
#         ci_lower = est - z*sd/np.sqrt(n)
#         ci_upper = est + z*sd/np.sqrt(n)
#         out['ci_lower'] = ci_lower
#         out['ci_upper'] = ci_upper
#
#     return out
#
#
# def est_cFPR(theta, data, A='A', R='R', outcome='phihat', ci='None'):
#     """Estimate the cFPR and the fairness gap."""
#     inf_func = data.groupby(A).apply(_est_cFPR)
#     pass


#### metrics: for a fixed post-processed predictor ####
def metrics(theta, data, A='A', R='R', outcome='phihat', ci=0.95, ci_scale='logit'):
    """Compute risk and fairness metrics wrt Y0.

    Args:
      data: data over which to compute the metrics.
      outcome:
        Using phihat as the outcome yields doubly robust estimators.
        Using muhat0 as the outcome yields plugin estimators.
        Using mu0 as the outcome yields (something close to) the true metrics.
      ci: Either None, in which case no CI is computed, or a value in (0, 1).
    """
    risk = est_risk(theta, data, A=A, R=R, outcome=outcome, ci=ci, ci_scale=ci_scale)
    cFPR = est_cFPR(theta, data, A=A, R=R, outcome=outcome, ci=ci, ci_scale=ci_scale)
    cFNR = est_cFNR(theta, data, A=A, R=R, outcome=outcome, ci=ci, ci_scale=ci_scale)

    out = pd.concat([risk, cFPR, cFNR])

    return out


def _metrics_to_df(res):
    """Convert multi-dim numpy array to a plottable DataFrame."""
    out = [pd.DataFrame(res[i, :, :]) for i in range(res.shape[0])]
    out = pd.concat(out, keys=list(range(len(out))), names=['mc_iter'])
    out = out.reset_index().drop(columns='level_1')
    out.columns = ['mc_iter', 'metric', 'value', 'ci_lower', 'ci_upper']

    return out


def metrics_to_df(res, n_arr, setting, data_val):
    out = [np.apply_along_axis(metrics, 1, rr['theta_arr'], data=data_val,
                               outcome='mu0') for rr in res]
    out = pd.concat([_metrics_to_df(arr) for arr in out], keys=n_arr)
    out = out.reset_index().drop(columns='level_1')
    out.columns = ['n', 'mc_iter', 'metric', 'value', 'ci_lower', 'ci_upper']
    out['setting'] = setting
    out['value'] = pd.to_numeric(out['value'])

    return out


def coverage(metrics_est, metrics_true, simplify=True):
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