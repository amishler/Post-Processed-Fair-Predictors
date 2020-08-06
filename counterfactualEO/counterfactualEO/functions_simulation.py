import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import confusion_matrix, classification_report
from .functions_estimation import *

def generate_data_pre(n, prob_A, beta_X, beta_D, beta_Y0, beta_Y1,
                      trunc_pi=0.975):
    """Generate sample data at time 0, including counterfactual outcomes.

    Here, D doesn't depend on R, since R hasn't been generated yet.
    """
    A = np.random.binomial(1, 0.3, size=(n, 1))
    X = np.random.normal(beta_X * A, size=(n, 4))
    AX = np.concatenate([A, X], axis=1)

    D_probs = expit(0.5 * np.dot(AX, beta_D)).clip(max=trunc_pi).reshape((n, 1))
    D = np.random.binomial(n=1, p=D_probs).reshape((n, 1))

    Y0_probs = expit(np.dot(AX, beta_Y0)).reshape((n, 1))
    Y0 = np.random.binomial(n=1, p=Y0_probs)
    Y1_probs = expit(np.dot(AX, beta_Y1)).reshape((n, 1))
    Y1 = np.random.binomial(n=1, p=Y1_probs)
    Y = D * Y1 + (1 - D) * Y0

    data = np.concatenate([AX, D, Y], axis=1)
    data = pd.DataFrame(data, columns=['A', 'X1', 'X2', 'X3', 'X4', 'D', 'Y'])

    return data


def generate_data_post(n, prob_A, beta_X, beta_D, beta_Y0, beta_Y1, model_R,
                       trunc_pi=0.975):
    """Generate sample data at time 1, including counterfactual outcomes.

    Here, D does depend on R. Y still depends on A, X, D.
    """
    A = np.random.binomial(1, 0.3, size=(n, 1))
    X = np.random.normal(beta_X * A, size=(n, 4))
    AX = np.concatenate([A, X], axis=1)

    R = model_R.predict(AX).reshape((n, 1))
    AXR = np.concatenate([A, X, R], axis=1)
    D_probs = expit(np.dot(AXR, beta_D)).clip(max=trunc_pi).reshape((n, 1))
    D = np.random.binomial(n=1, p=D_probs).reshape((n, 1))

    Y0_probs = expit(np.dot(AX, beta_Y0)).reshape((n, 1))
    Y0 = np.random.binomial(n=1, p=Y0_probs)
    Y1_probs = expit(np.dot(AX, beta_Y1)).reshape((n, 1))
    Y1 = np.random.binomial(n=1, p=Y1_probs)
    Y = D * Y1 + (1 - D) * Y0

    ## Nuisance parameter phi
    phi = (1 - D) / (1 - D_probs) * (Y - Y0_probs) + Y0_probs

    data = np.concatenate([AXR, D, Y0, Y1, Y, D_probs, Y0_probs, phi], axis=1)
    data = pd.DataFrame(data, columns=['A', 'X1', 'X2', 'X3', 'X4', 'R', 'D',
                                       'Y0', 'Y1', 'Y', 'pi', 'mu0', 'phi'])

    return data


def generate_data_post_noisy(n, noise_coef, prob_A, beta_X, beta_D, beta_Y0,
                             beta_Y1,
                             model_R, trunc_pi=0.975):
    """Generate sample data at time 1, including counterfactual outcomes.

    Here, D does depend on R. Y still depends on A, X, D.

    Args:
      noise_coef: multiplier for the noise.
    """
    A = np.random.binomial(1, 0.3, size=(n, 1))
    X = np.random.normal(beta_X * A, size=(n, 4))
    AX = np.concatenate([A, X], axis=1)

    R = model_R.predict(AX).reshape((n, 1))
    AXR = np.concatenate([A, X, R], axis=1)
    D_probs = expit(np.dot(AXR, beta_D)).clip(max=trunc_pi).reshape((n, 1))
    D = np.random.binomial(n=1, p=D_probs).reshape((n, 1))

    Y0_probs = expit(np.dot(AX, beta_Y0)).reshape((n, 1))
    Y0 = np.random.binomial(n=1, p=Y0_probs)
    Y1_probs = expit(np.dot(AX, beta_Y1)).reshape((n, 1))
    Y1 = np.random.binomial(n=1, p=Y1_probs)
    Y = D * Y1 + (1 - D) * Y0

    ## Nuisance parameter phi
    phi = (1 - D) / (1 - D_probs) * (Y - Y0_probs) + Y0_probs

    ## Noisy "estimates" of D_probs (pihat), Y0_probs (muhat0), and phi (phihat)
    pihat_logit = np.dot(AXR, beta_D) + noise_coef * np.random.normal(
        size=n) / n ** 0.26
    pihat = expit(pihat_logit).clip(max=trunc_pi).reshape((n, 1))
    muhat0_logit = np.dot(AX, beta_Y0) + noise_coef * np.random.normal(
        size=n) / n ** 0.26
    muhat0 = expit(muhat0_logit).reshape((n, 1))
    phihat = (1 - D) / (1 - pihat) * (Y - muhat0) + muhat0

    data = np.concatenate([AXR, D, Y0, Y1, Y, D_probs, pihat, Y0_probs, muhat0,
                           phi, phihat], axis=1)
    data = pd.DataFrame(data, columns=['A', 'X1', 'X2', 'X3', 'X4', 'R', 'D',
                                       'Y0', 'Y1', 'Y', 'pi', 'pihat', 'mu0',
                                       'muhat0', 'phi', 'phihat'])

    return data


def check_data_post(data, A='A', D='D', Y='Y', Y0='Y0', pi='pi'):
    """Make sure the input distribution looks reasonable."""
    print('===============================================================================================')
    print('Cross-tabs for sensitive feature A, decision D, potential outcome Y0, and observable outcome Y.')
    print('===============================================================================================')

    print('---------------')
    print(pd.crosstab(data[A], data[D]))
    print('---------------')
    print(pd.crosstab(data[A], data[Y0]))
    print('---------------')
    print(pd.crosstab(data['A'], data['Y']))
    print('---------------')
    print('\nHow often Y = Y0:', np.mean(data['Y0'] == data['Y']), '\n')

    print('===============================')
    print('Properties of input predictor R')
    print('===============================\n')

    print('-------------------------')
    print('Confusion matrix for Y, R')
    print('-------------------------')
    print(confusion_matrix(data['Y'], data['R'])/data.shape[0], '\n')

    print('------------------------------')
    print('Classification report for Y, R')
    print('------------------------------')
    print(classification_report(data['Y'], data['R'], target_names=['Y = 0', 'Y = 1']), '\n')

    print('-------------------------------')
    print('Classification report for Y0, R')
    print('-------------------------------')
    print(classification_report(data['Y0'], data['R'], target_names=['Y0 = 0', 'Y0 = 1']), '\n')

    print('------------------------------------------------------')
    print('Distribution of propensity scores by sensitive feature')
    print('------------------------------------------------------')
    data.groupby(A)[pi].hist()


def get_optimal(data_params, epsilon_pos, epsilon_neg, n=50000):
    """Get the optimal fair predictor."""

    ## Generate data
    data_train = generate_data_post(n, **data_params)

    ## Get 'true' loss and fairness constraints
    coefs_obj = risk_coefs(data_train, 'A', 'R', 'mu0')
    coefs_pos, coefs_neg = fairness_coefs(data_train, 'A', 'R', 'mu0')

    ## Get best derived predictor
    theta = optimize(coefs_obj, coefs_pos, coefs_neg, epsilon_pos, epsilon_neg)

    out = {'theta': theta, 'obj': coefs_obj, 'pos': coefs_pos, 'neg': coefs_neg}

    return out


def dist_to_ref(arr, vec):
    """Get L2 distance between a vector and each column of an array."""
    out = np.apply_along_axis(lambda col: np.linalg.norm(col - vec), 1, arr)
    return out


def add_noise_logit(n_arr, mc_reps, data_params, obj_true, pos_true, neg_true,
                    noise_coef, trunc_pi=0.975, verbose=True):
    """Add noise to logit-transformed pi and mu0.

    This ensures the noise doesn't push the estimates outside [0, 1].
    """
    n_rows = len(n_arr) * mc_reps
    obj_arr = np.zeros((n_rows, 4))
    pos_arr = np.zeros((n_rows, 4))
    neg_arr = np.zeros((n_rows, 4))

    for j, n in enumerate(n_arr):
        print('Sample size {}:'.format(n))
        for i in range(j * mc_reps, j * mc_reps + mc_reps):
            if verbose and (i % 10 == 0):
                print("...Round {}".format(i))

                ## Generate data
            data_train = generate_data_post_noisy(n, noise_coef, **data_params)

            ## Compute the LP coefficients
            obj = risk_coefs(data_train, A=A, R=R, outcome='phihat')
            obj_arr[i, :] = obj
            fair_pos, fair_neg = fairness_coefs(data_train, A=A, R=R,
                                                outcome='phihat')
            pos_arr[i, :] = fair_pos
            neg_arr[i, :] = fair_neg

    ## Get total L2 distance of noisy coefficients from true coefficients
    dist_obj = dist_to_ref(obj_arr, obj_true)
    dist_pos = dist_to_ref(pos_arr, pos_true)
    dist_neg = dist_to_ref(neg_arr, neg_true)

    #     n_col = np.array([nn for nn in n_arr for k in range(mc_reps)]).reshape((n_rows, 1))
    n_col = pd.Series([nn for nn in n_arr for k in range(mc_reps)] * 3)
    id_col = pd.Series(
        [mm for mm in ['obj', 'pos', 'neg'] for k in range(n_rows)])
    dist_col = pd.Series(np.concatenate([dist_obj, dist_pos, dist_neg]))

    comb = pd.DataFrame(np.concatenate([obj_arr, pos_arr, neg_arr], axis=0))
    out = pd.concat([n_col, id_col, comb, dist_col], axis=1)
    #     out = pd.concat([out, dist_col], axis = 1)
    out.columns = ['n', 'id', 'comp0', 'comp1', 'comp2', 'comp3', 'L2']

    return out


def add_noise_expit(n_arr, mc_reps, data_params, obj_true, pos_true, neg_true,
                    trunc_pi=0.975, verbose=True):
    """Add noise to nuisance parameter values and see how it affects the LP."""
    n_rows = len(n_arr) * mc_reps
    obj_arr = np.zeros((n_rows, 4))
    pos_arr = np.zeros((n_rows, 4))
    neg_arr = np.zeros((n_rows, 4))

    for j, n in enumerate(n_arr):
        print('Sample size {}:'.format(n))
        for i in range(j * mc_reps, j * mc_reps + mc_reps):
            if verbose and (i % 10 == 0):
                print("...Round {}".format(i))

                ## Generate data
            data_train = generate_data_post(n, **data_params)
            mu0_noise = np.random.uniform(-1, 1, n) / n ** (0.26) * 10
            pi_noise = np.random.uniform(-1, 1, n) / n ** (0.26) * 10
            data_train['muhat0'] = (data_train['mu0'] + mu0_noise).clip(0, 1)
            data_train['pihat'] = (data_train['pi'] + pi_noise).clip(0,
                                                                     trunc_pi)
            data_train['phihat'] = (1 - data_train['D']) / (
                        1 - data_train['pihat']) * \
                                   (data_train['Y'] - data_train['muhat0']) + \
                                   data_train['muhat0']

            ## Compute the LP coefficients
            obj = risk_coefs(data_train, A=A, R=R, outcome='phihat')
            obj_arr[i, :] = obj
            fair_pos, fair_neg = fairness_coefs(data_train, A=A, R=R,
                                                outcome='phihat')
            pos_arr[i, :] = fair_pos
            neg_arr[i, :] = fair_neg

    ## Get total L2 distance of noisy coefficients from true coefficients
    dist_obj = dist_to_ref(obj_arr, obj_true)
    dist_pos = dist_to_ref(pos_arr, pos_true)
    dist_neg = dist_to_ref(neg_arr, neg_true)

    #     n_col = np.array([nn for nn in n_arr for k in range(mc_reps)]).reshape((n_rows, 1))
    n_col = pd.Series([nn for nn in n_arr for k in range(mc_reps)] * 3)
    id_col = pd.Series(
        [mm for mm in ['obj', 'pos', 'neg'] for k in range(n_rows)])
    dist_col = pd.Series(np.concatenate([dist_obj, dist_pos, dist_neg]))

    comb = pd.DataFrame(np.concatenate([obj_arr, pos_arr, neg_arr], axis=0))
    out = pd.concat([n_col, id_col, comb, dist_col], axis=1)
    #     out = pd.concat([out, dist_col], axis = 1)
    out.columns = ['n', 'id', 'comp0', 'comp1', 'comp2', 'comp3', 'L2']

    return out, data_train


def sim_theta(n, mc_reps, noise_coef, data_params, epsilon_pos, epsilon_neg,
              A='A', R='R', outcome='phihat', trunc_pi=0.975, verbose=False):
    """Simulate by adding noise to true nuisance parameters.

    ...as opposed to estimating the nuisance parameters.

    Args:

      outcome: 'phihat' yields doubly robust estimators for the linear program.
        'muhat0' yields a singly robust plugin estimator.
    """
    print('Simulating theta-hat for sample size {}'.format(n))
    theta_arr = np.zeros((mc_reps, 4))

    for i in range(mc_reps):
        if verbose and (i % 10 == 0):
            print("...Round {}".format(i))

            ## Generate data, compute and solve the LP
        data_train = generate_data_post_noisy(n, noise_coef, **data_params)
        obj = risk_coefs(data_train, A=A, R=R, outcome=outcome)
        fair_pos, fair_neg = fairness_coefs(data_train, A=A, R=R,
                                            outcome=outcome)
        theta = optimize(obj, fair_pos, fair_neg, epsilon_pos, epsilon_neg)
        theta_arr[i, :] = theta

    out = {'n': n, 'theta_arr': theta_arr}

    return out


def _sim_task2(theta, noise_coef, n, mc_reps, data_params, outcome='phihat',
               trunc_pi=0.975, ci_scale='logit', verbose=False):
    """Simulate estimating the risk and fairness properties of a fixed predictor."""
    print('Estimating metrics for sample size {}: {} reps'.format(n, mc_reps))

    res = [None] * mc_reps

    for i in range(mc_reps):
        if verbose and (i % 10 == 0):
            print("...Round {}".format(i))

        ## Generate data, estimate metrics
        data_val = generate_data_post_noisy(n, noise_coef, **data_params)
        res[i] = metrics(theta, data_val, outcome=outcome, ci_scale=ci_scale)

    out = pd.concat(res, keys=list(range(mc_reps)))
    out = out.reset_index().drop(columns='level_1')
    out.columns = ['mc_iter', 'metric', 'value', 'ci_lower', 'ci_upper']

    return out


def sim_task2(theta, noise_coef, n_arr, mc_reps, data_params, trunc_pi=0.975,
              outcome='phihat', ci_scale='logit', verbose=False):
    metrics_list = [_sim_task2(theta, noise_coef, n, mc_reps,
                               data_params, outcome=outcome,
                               ci_scale=ci_scale, verbose=verbose) for n in n_arr]
    metrics_est = pd.concat(metrics_list, keys=n_arr)
    metrics_est = metrics_est.reset_index().drop(columns='level_1')
    metrics_est.columns = ['n', 'mc_iter', 'metric', 'value', 'ci_lower', 'ci_upper']

    return metrics_est


def simulate_true(n, data_params):
    """Get the 'true' best fair predictor, given access to true values of the
    nuisance parameters."""
    ## Generate data
    data_opt = generate_data_post(n, **data_params)

    ## Get empirical coefficients for loss and fairness constraints
    obj = risk_coefs(test, 'A', 'R', 'mu0')
    fair_pos, fair_neg = fairness_coefs(test, 'A', 'R', 'Y0')

    ## Get best derived predictor
    theta = optimize(loss_true, coefs_pos, coefs_neg, epsilon_pos, epsilon_neg)

    ## Get metrics of best derived predictor
    data_val = generate_data_post(n, **data_params)
    metrics = metrics(theta, data_val, 'A', 'R', 'Y0')
    out = {'theta': theta, 'risk_coefs': obj,
           'fairness_coefs_pos': fair_pos, 'fairness_coefs_neg': fair_neg,
           'metrics': metrics}

    return out





