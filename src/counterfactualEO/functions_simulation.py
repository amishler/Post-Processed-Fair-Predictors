"""
functions_simulations.py

This module provides tools for simulating data and evaluating counterfactually
fair predictors in experimental settings. It includes functions for:

- Generating synthetic data before and after deployment of a risk model,
  including true and noisy nuisance parameters (propensity scores and outcome models)
- Computing oracle and noisy estimates of linear program (LP) coefficients
  for fairness-constrained optimization
- Evaluating the performance and fairness of derived predictors
- Studying how nuisance estimation error and sample size affect fairness-constrained
  decision rules (theta) and their evaluation

The module supports two core experimental tasks:
1. Estimating a counterfactually fair predictor (theta-hat) under noise
2. Evaluating a fixed predictor (theta) across multiple sample sizes
"""
from itertools import product
import warnings

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from counterfactualEO.functions_estimation import (
    optimize, risk_coefs, fairness_coefs
)
from counterfactualEO.functions_evaluation import metrics_post


def generate_data_pre(n, prob_A, beta_X, beta_D, beta_Y0, beta_Y1,
                      trunc_pi=0.975):
    """
    Generate synthetic data at time 0, where treatment decisions are made 
    without a risk score.

    The treatment D is based only on the features (A, X), where A is the
    sensitive feature and X represents 4 additional features. The outcome Y is 
    generated from potential (aka counterfactual) outcomes Y0 and Y1.

    Args:
        n (int): Number of samples to generate.
        prob_A (float): Probability of A = 1 (sensitive attribute).
        beta_X (array): Coefficients for generating features X from A.
        beta_D (array): Coefficients for generating D (treatment, aka decision) 
            from (A, X).
        beta_Y0 (array): Coefficients for generating Y0, the potential outcome 
            under the baseline treatment level D = 0.
        beta_Y1 (array): Coefficients for generating Y1, the potential outcome
            under the baseline treatment level D = 1.
        trunc_pi (float): Maximum value for the propensity score for the
            propensity score for treatment D (to prevent instability).

    Returns:
        pd.DataFrame: DataFrame with columns ['A', 'X1', 'X2', 'X3', 'X4', 'D', 'Y'].
    """
    A = np.random.binomial(1, prob_A, size=(n, 1))
    X = np.random.normal(beta_X * A, size=(n, 4))
    AX = np.concatenate([A, X], axis=1)

    D_probs = expit(0.5 * np.dot(AX, beta_D)).clip(max=trunc_pi).reshape((n, 1))
    D = np.random.binomial(n=1, p=D_probs).reshape((n, 1))

    Y0_probs = expit(np.dot(AX, beta_Y0)).reshape((n, 1))
    Y0 = np.random.binomial(n=1, p=Y0_probs)
    Y1_probs = expit(np.dot(AX, beta_Y1)).reshape((n, 1))
    Y1 = np.random.binomial(n=1, p=Y1_probs)
    Y = D * Y1 + (1 - D) * Y0

    ## Nuisance parameter phi
    phi = (1 - D) / (1 - D_probs) * (Y - Y0_probs) + Y0_probs

    data = np.concatenate([AX, D, Y0, Y1, Y, D_probs, Y0_probs, phi], axis=1)
    data = pd.DataFrame(data, columns=['A', 'X1', 'X2', 'X3', 'X4', 'D',
                                       'Y0', 'Y1', 'Y', 'pi', 'mu0', 'phi'])

    return data


def generate_data_post(n, prob_A, beta_X, beta_D, beta_Y0, beta_Y1, model_R,
                       trunc_pi=0.975):
    """
    Generate synthetic data at time 1, where treatment depends on a risk score.

    Treatment D is assigned based on (A, X, R), where R is a prediction from a 
    pretrained model. Counterfactual outcomes Y0 and Y1 are generated, and a 
    doubly robust pseudo-outcome phi is computed.

    Args:
        n (int): Number of samples to generate.
        prob_A (float): Probability of A = 1.
        beta_X (array): Coefficients for generating X from A.
        beta_D (array): Coefficients for generating D from (A, X, R).
        beta_Y0 (array): Coefficients for generating Y0.
        beta_Y1 (array): Coefficients for generating Y1.
        model_R: A risk model with a `.predict()` method.
        trunc_pi (float): Maximum value for D's propensity score.

    Returns:
        pd.DataFrame: Full dataset with counterfactual and nuisance columns:
            ['A', 'X1', ..., 'X4', 'R', 'D', 'Y0', 'Y1', 'Y', 'pi', 'mu0', 'phi'].
    """
    A = np.random.binomial(1, prob_A, size=(n, 1))
    X = np.random.normal(beta_X * A, size=(n, 4))
    AX = np.concatenate([A, X], axis=1)
    AX = pd.DataFrame(AX, columns=['A', 'X1', 'X2', 'X3', 'X4'])

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
                             beta_Y1, model_R, trunc_pi=0.975):
    """
    Generate synthetic data at time 1 with added noise to simulate nuisance 
    estimation error.

    This function simulates estimation error by adding noise to the 
    logit-transformed nuisance parameters (D's propensity, Y0 outcome model), 
    and computes a noisy doubly robust pseudo-outcome (phihat).

    Args:
        n (int): Number of samples.
        noise_coef (float): Multiplier for controlling noise level.
        prob_A (float): Probability of A = 1.
        beta_X (array): Coefficients for generating X from A.
        beta_D (array): Coefficients for generating D from (A, X, R).
        beta_Y0 (array): Coefficients for Y0 model.
        beta_Y1 (array): Coefficients for Y1 model.
        model_R: Risk model with `.predict()` method.
        trunc_pi (float): Truncation value for propensity score to avoid 
            instability.

    Returns:
        pd.DataFrame: Dataset with true and noisy nuisance estimates:
            ['A', 'X1', ..., 'X4', 'R', 'D', 'Y0', 'Y1', 'Y',
             'pi', 'pihat', 'mu0', 'muhat0', 'phi', 'phihat'].
    """
    A = np.random.binomial(1, prob_A, size=(n, 1))
    X = np.random.normal(beta_X * A, size=(n, 4))
    AX = np.concatenate([A, X], axis=1)
    AX = pd.DataFrame(AX, columns=['A', 'X1', 'X2', 'X3', 'X4'])

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


def check_data_post(data, A='A', D='D', R='R', Y='Y', Y0='Y0', pi='pi'):
    """
    Print diagnostic statistics to inspect distribution and validity of 
    post-intervention data.

    This function is useful for sanity checks when working with synthetic data. 
    It prints:
    - Crosstabs of A vs. D, Y0, and Y
    - Frequency of Y = Y0
    - Confusion matrices and classification reports for R vs. Y and Y0
    - Histograms of propensity scores grouped by A

    Args:
        data (pd.DataFrame): The post-intervention dataset.
        A (str): Name of the sensitive attribute column.
        D (str): Name of the decision column.
        R (str): Name of the prediction column.
        Y (str): Name of the observed outcome column.
        Y0 (str): Name of the counterfactual outcome under D = 0.
        pi (str): Name of the propensity score column.
    """
    print('===============================================================================================')
    print('Cross-tabs for sensitive feature A, decision D, potential outcome Y0, and observable outcome Y.')
    print('===============================================================================================')

    print('---------------')
    print(pd.crosstab(data[A], data[D]))
    print('---------------')
    print(pd.crosstab(data[A], data[Y0]))
    print('---------------')
    print(pd.crosstab(data[A], data[Y]))
    print('---------------')
    print('\nHow often Y = Y0:', np.mean(data[Y0] == data[Y]), '\n')

    print('===============================')
    print('Properties of input predictor R')
    print('===============================\n')

    print('-------------------------')
    print('Confusion matrix for Y, R')
    print('-------------------------')
    print(confusion_matrix(data[Y], data[R])/data.shape[0], '\n')

    print('------------------------------')
    print('Classification report for Y, R')
    print('------------------------------')
    print(classification_report(data[Y], data[R], target_names=['Y = 0', 'Y = 1']), '\n')

    print('-------------------------------')
    print('Classification report for Y0, R')
    print('-------------------------------')
    print(classification_report(data[Y0], data[R], target_names=['Y0 = 0', 'Y0 = 1']), '\n')

    print('------------------------------------------------------')
    print('Distribution of propensity scores by sensitive feature')
    print('------------------------------------------------------')
    data.groupby(A)[pi].hist()


def dist_to_ref(arr, vec):
    """
    Compute the L2 distance between each row of a matrix and a reference vector.

    Useful for comparing estimated LP coefficients to ground truth.

    Args:
        arr (np.ndarray): 2D array where each row is a vector to compare.
        vec (np.ndarray): 1D reference vector.

    Returns:
        np.ndarray: Array of L2 distances.
    """
    out = np.apply_along_axis(lambda col: np.linalg.norm(col - vec), 1, arr)
    return out


def add_noise_logit(n_arr, mc_reps, data_params, obj_true, pos_true, neg_true,
                    noise_coef, verbose=True):
    """
    Add noise to the logit-transformed nuisance parameters (pi, mu0) and compute 
    effect on LP coefficients.

    The noise is added in logit space to ensure values stay within [0, 1] after 
    applying expit to transform pi and mu0 back to probability space.
    This function simulates the effect of estimation error in the nuisance
    parameters on the LP coefficients.

    Args:
        n_arr (list): List of sample sizes.
        mc_reps (int): Number of Monte Carlo repetitions for each n.
        data_params (dict): Parameters for data generation.
        obj_true (np.ndarray): True objective (risk) coefficients.
        pos_true (np.ndarray): True FPR constraint coefficients.
        neg_true (np.ndarray): True FNR constraint coefficients.
        noise_coef (float): Scale of the noise added to logits.
        trunc_pi (float): Maximum value for propensity scores.
        verbose (bool): Whether to print progress.

    Returns:
        pd.DataFrame: DataFrame containing noisy estimates, L2 distances, and 
            sample sizes.
    """
    n_rows = len(n_arr) * mc_reps
    obj_arr = np.zeros((n_rows, 4))
    pos_arr = np.zeros((n_rows, 4))
    neg_arr = np.zeros((n_rows, 4))

    for j, n in enumerate(n_arr):
        if verbose:
            print('Sample size {}:'.format(n))
        for i in range(j * mc_reps, j * mc_reps + mc_reps):
            if verbose and (i % 10 == 0):
                print("...Round {}".format(i))

            ## Generate data
            data_train = generate_data_post_noisy(n, noise_coef, **data_params)

            ## Compute the LP coefficients
            obj = risk_coefs(data_train, A='A', R='R', outcome='phihat')
            obj_arr[i, :] = obj
            fair_pos, fair_neg = fairness_coefs(data_train, A='A', R='R',
                                                outcome='phihat')
            pos_arr[i, :] = fair_pos
            neg_arr[i, :] = fair_neg

    ## Get total L2 distance of noisy coefficients from true coefficients
    dist_obj = dist_to_ref(obj_arr, obj_true)
    dist_pos = dist_to_ref(pos_arr, pos_true)
    dist_neg = dist_to_ref(neg_arr, neg_true)

    n_col = pd.Series([nn for nn in n_arr for k in range(mc_reps)] * 3)
    id_col = pd.Series(
        [mm for mm in ['obj', 'pos', 'neg'] for k in range(n_rows)])
    dist_col = pd.Series(np.concatenate([dist_obj, dist_pos, dist_neg]))

    comb = pd.DataFrame(np.concatenate([obj_arr, pos_arr, neg_arr], axis=0))
    out = pd.concat([n_col, id_col, comb, dist_col], axis=1)
    out.columns = ['n', 'id', 'comp0', 'comp1', 'comp2', 'comp3', 'L2']

    return out


def add_noise_expit(n_arr, mc_reps, data_params, obj_true, pos_true, neg_true,
                    noise_coef, trunc_pi=0.975, verbose=True):
    """
    Add noise directly to nuisance parameters (in probability space) and compute 
    effect on LP coefficients.

    Unlike `add_noise_logit`, this adds noise directly to mu0 and pi estimates,
    which must then be clipped to [0, 1] to ensure valid probabilities.
    This function simulates the effect of estimation error in the nuisance
    parameters on the LP coefficients.

    Args:
        n_arr (list): List of sample sizes.
        mc_reps (int): Number of Monte Carlo repetitions per sample size.
        data_params (dict): Parameters for data generation using
          generate_data_post().
        obj_true, pos_true, neg_true (np.ndarray): Ground truth LP coefficients.
        trunc_pi (float): Truncation value for pi.
        verbose (bool): Whether to print progress updates.

    Returns:
        pd.DataFrame: Summary dataframe of results. Gives the LP coefficients
         estimated from the noisy versions of the nuisance parameters as well as 
         the total L2 distances of each set of 4 coefficients (the objective
         function, the fairness constraints for epsilon_pos, and the fairness
         constraints for epsilon_neg) from the true optimal values.
    """
    n_rows = len(n_arr) * mc_reps
    obj_arr = np.zeros((n_rows, 4))
    pos_arr = np.zeros((n_rows, 4))
    neg_arr = np.zeros((n_rows, 4))

    for j, n in enumerate(n_arr):
        if verbose:
            print('Sample size {}:'.format(n))
        for i in range(j * mc_reps, j * mc_reps + mc_reps):
            if verbose and (i % 10 == 0):
                print("...Round {}".format(i))

            ## Generate data
            data_train = generate_data_post(n, **data_params)
            mu0_noise = np.random.uniform(-1, 1, n) * noise_coef / n ** (0.26)
            pi_noise = np.random.uniform(-1, 1, n) * noise_coef / n ** (0.26)
            data_train['muhat0'] = (data_train['mu0'] + mu0_noise).clip(0, 1)
            data_train['pihat'] = (data_train['pi'] + pi_noise).clip(0,
                                                                     trunc_pi)
            data_train['phihat'] = (1 - data_train['D']) / (
                        1 - data_train['pihat']) * \
                                   (data_train['Y'] - data_train['muhat0']) + \
                                   data_train['muhat0']

            ## Compute the LP coefficients
            obj = risk_coefs(data_train, A='A', R='R', outcome='phihat')
            obj_arr[i, :] = obj
            fair_pos, fair_neg = fairness_coefs(data_train, A='A', R='R',
                                                outcome='phihat')
            pos_arr[i, :] = fair_pos
            neg_arr[i, :] = fair_neg

    ## Get total L2 distance of noisy coefficients from true coefficients
    dist_obj = dist_to_ref(obj_arr, obj_true)
    dist_pos = dist_to_ref(pos_arr, pos_true)
    dist_neg = dist_to_ref(neg_arr, neg_true)

    n_col = pd.Series([nn for nn in n_arr for k in range(mc_reps)] * 3)
    id_col = pd.Series(
        [mm for mm in ['obj', 'pos', 'neg'] for k in range(n_rows)])
    dist_col = pd.Series(np.concatenate([dist_obj, dist_pos, dist_neg]))

    comb = pd.DataFrame(np.concatenate([obj_arr, pos_arr, neg_arr], axis=0))
    out = pd.concat([n_col, id_col, comb, dist_col], axis=1)
    out.columns = ['n', 'id', 'coef1', 'coef2', 'coef3', 'coef4', 'L2']

    return out


def simulate_true(n, data_params, epsilon_pos, epsilon_neg, ci=0.95, ci_scale='logit'):
    """
    Compute the optimal fair predictor using true nuisance parameters, and evaluate it.

    This function generates synthetic data, uses the true (oracle) values of mu0 
    and Y0 to solve the fairness-constrained LP, and then evaluates the 
    resulting predictor indexed by theta on a separate validation dataset.

    Args:
        n (int): Number of samples to generate for both optimization and 
            evaluation.
        data_params (dict): Parameters for generate_data_post.
        epsilon_pos (float): Tolerance for FPR fairness constraint.
        epsilon_neg (float): Tolerance for FNR fairness constraint.

    Returns:
        dict: Dictionary with keys:
            - 'theta': Optimal decision rule.
            - 'risk_coefs': Objective coefficients from mu0.
            - 'fairness_coefs_pos': FPR constraint coefficients from Y0.
            - 'fairness_coefs_neg': FNR constraint coefficients from Y0.
            - 'metrics': Evaluation metrics from separate dataset.
    """
    # Generate optimization dataset
    data_opt = generate_data_post(n, **data_params)

    ## Get empirical coefficients for loss and fairness constraints
    risk_coefs_ = risk_coefs(data_opt, A='A', R='R', outcome='mu0')
    fair_pos, fair_neg = fairness_coefs(data_opt, A='A', R='R', outcome='Y0')

    # Solve LP to get best derived predictor
    theta = optimize(risk_coefs_, fair_pos, fair_neg, epsilon_pos, epsilon_neg)

    # Evaluate best derived predictor on a new, independent dataset
    data_val = generate_data_post(n, **data_params)
    evals = metrics_post(theta, data_val, A='A', R='R', outcome='Y0', 
                         ci=ci, ci_scale=ci_scale)

    return {
        'theta': theta,
        'risk_coefs': risk_coefs_,
        'fairness_coefs_pos': fair_pos,
        'fairness_coefs_neg': fair_neg,
        'metrics': evals,
        'epsilon_pos': epsilon_pos,
        'epsilon_neg': epsilon_neg
    }


def simulate_task1(n, mc_reps, noise_coef, data_params, epsilon_pos, epsilon_neg,
              A='A', R='R', outcome='phihat', verbose=False, n_jobs=-1):
    """
    Simulate sampling variability in theta-hat under noisy nuisance parameters.

    This function performs repeated data generation, nuisance perturbation, and
    LP solving to study variability in the estimated decision rule theta.

    Args:
        n (int): Sample size per repetition.
        mc_reps (int): Number of Monte Carlo repetitions.
        noise_coef (float): Magnitude of noise added to nuisance parameters.
        data_params (dict): Parameters for data generation.
        epsilon_pos (float): Allowed deviation in FPR fairness constraint.
        epsilon_neg (float): Allowed deviation in FNR fairness constraint.
        A (str): Sensitive attribute column name.
        R (str): Risk score column name.
        outcome (str): Outcome used in optimization (e.g. 'phihat' or 'muhat0').
        verbose (bool): Whether to print progress.
        n_jobs (int): Number of parallel jobs (default: -1 for all available cores).

    Returns:
        dict: Dictionary with sample size and theta estimates across repetitions.
    """
    print(f'Simulating theta-hat for sample size {n} with {mc_reps} repetitions...')

    def single_simulation(i):
        if verbose and (i % 10 == 0):
            print(f"...Round {i}")
        data_train = generate_data_post_noisy(n, noise_coef, **data_params)
        obj = risk_coefs(data_train, A=A, R=R, outcome=outcome)
        fair_pos, fair_neg = fairness_coefs(data_train, A=A, R=R, outcome=outcome)
        theta = optimize(obj, fair_pos, fair_neg, epsilon_pos, epsilon_neg)
        return theta

    results = Parallel(n_jobs=n_jobs)(
        delayed(single_simulation)(i) for i in range(mc_reps)
    )

    theta_arr = np.vstack(results)
    return {
        'n': n, 
        'theta_arr': theta_arr, 
        'epsilon_pos': epsilon_pos,
        'epsilon_neg': epsilon_neg
    }


def _eval_one_theta(n_val, mc_iter, theta_row, epsilon_pos, epsilon_neg, 
                    data_val, ci, ci_scale='logit'):
    """
    Evaluate a single theta and return its metrics as a DataFrame row.
    """
    df = metrics_post(theta_row, data_val, A='A', R='R', outcome='mu0', ci=ci, 
                      ci_scale=ci_scale)
    df.insert(0, 'mc_iter', mc_iter)
    df.insert(0, 'n', n_val)
    df.insert(0, 'epsilon_neg', epsilon_neg)
    df.insert(0, 'epsilon_pos', epsilon_pos)
    return df


def simulate_task1_metrics_to_df(res, n_arr, setting, data_val, ci=0.95, 
                                 ci_scale='logit', n_jobs=-1):
    """
    Compute metrics estimates for all the thetas generated by `simulate_task1`.
    
    Fully parallelized version: evaluates metrics for each theta separately.

    Args:
        res (list): List of sim_theta outputs (each with 'theta_arr', 'epsilon_pos', 'epsilon_neg').
        n_arr (list): Corresponding sample sizes.
        setting (str): Experiment label to tag results.
        data_val (pd.DataFrame): Validation data for evaluation.
        ci (float): Confidence interval level.
        n_jobs (int): Number of parallel jobs (-1 = all cores).

    Returns:
        pd.DataFrame: Combined long-format metrics with confidence intervals.
    """
    jobs = []
    for rr, n_val in zip(res, n_arr):
        epsilon_pos = rr.get('epsilon_pos')
        epsilon_neg = rr.get('epsilon_neg')
        for mc_iter, theta_row in enumerate(rr['theta_arr']):
            jobs.append((n_val, mc_iter, theta_row, epsilon_pos, epsilon_neg))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_one_theta)(n_val, mc_iter, theta_row, epsilon_pos, 
                                 epsilon_neg, data_val, ci, ci_scale)
        for n_val, mc_iter, theta_row, epsilon_pos, epsilon_neg in tqdm(jobs, desc="Evaluating metrics for given values of theta")
    )

    df = pd.concat(results, ignore_index=True)
    df['setting'] = setting
    df['value'] = pd.to_numeric(df['value'])

    return df


def simulate_task2(theta, noise_coef, n_arr, mc_reps, data_params,
              outcome='phihat', ci=0.95, ci_scale='logit', verbose=False, n_jobs=-1):
    """
    Parallel simulation of evaluating a fixed predictor under sampling variability.

    For each sample size in `n_arr`, this function runs `mc_reps` Monte Carlo
    evaluations, where each one generates a new dataset, adds noise to the
    nuisance estimates, and computes evaluation metrics for a fixed predictor.

    Args:
        theta (np.ndarray): Fixed decision vector (e.g., from an LP solution).
        noise_coef (float): Magnitude of nuisance parameter noise.
        n_arr (List[int]): List of sample sizes to evaluate.
        mc_reps (int): Number of Monte Carlo repetitions per sample size.
        data_params (dict): Parameters for generating synthetic data.
        outcome (str): Outcome column used for evaluation ('phihat' or 'muhat0').
        ci_scale (str): Confidence interval scale ('logit' or 'expit').
        verbose (bool): Whether to display progress during simulation.
        n_jobs (int): Number of parallel jobs to run. Default (-1) uses all cores.

    Returns:
        pd.DataFrame: Long-format metric estimates with columns:
                      ['n', 'mc_iter', 'metric', 'value', 'ci_lower', 'ci_upper']
    """
    def simulate_one(n, i):
        if verbose and (i % 10 == 0):
            print(f"[n={n}] Simulating run {i}")
        data_val = generate_data_post_noisy(n, noise_coef, **data_params)
        result = metrics_post(theta, data_val, outcome=outcome, ci=ci, ci_scale=ci_scale)
        result.insert(0, 'mc_iter', i)
        result.insert(0, 'n', n)
        return result

    tasks = [(n, i) for n in n_arr for i in range(mc_reps)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_one)(n, i) for n, i in tasks
    )

    return pd.concat(results).reset_index(drop=True)


def simulate_performance_tradeoff(
    n, data_params, epsilon_pos_arr, epsilon_neg_arr, n_jobs=-1
):
    """
    Calculate accuracy-fairness tradeoff for the optimal input predictor
    vs. the Bayes-optimal predictor across multiple (ε_pos, ε_neg) combinations.
    
    Optimal input predictor refers to the Bayes-optimal predictor of Y0, not Y.
    This function is similar to simulate_true(), but instead of generating the
    data using generate_data_post(), which takes a model for the predictor R,
    this generates data using generate_data_pre() and then sets R to be the
    Bayes-optimal predictor based on the true values of mu0. 

    Args:
        n (int): Sample size.
        data_params (dict): Parameters to pass to `generate_data_pre`.
        epsilon_pos_arr (list or np.ndarray): Candidate ε_pos values.
        epsilon_neg_arr (list or np.ndarray): Candidate ε_neg values.
        n_jobs (int): Number of parallel jobs (default: -1 = all cores).

    Returns:
        pd.DataFrame: Long-format metric results with ε values annotated.
    """
    # Step 1: Validate and filter epsilon combinations
    valid_combos = []
    for eps_pos, eps_neg in product(epsilon_pos_arr, epsilon_neg_arr):
        if not (0 <= eps_pos <= 1):
            warnings.warn(f"Ignoring invalid ε_pos = {eps_pos}")
            continue
        if not (0 <= eps_neg <= 1):
            warnings.warn(f"Ignoring invalid ε_neg = {eps_neg}")
            continue
        valid_combos.append((eps_pos, eps_neg))

    if not valid_combos:
        raise ValueError("No valid (ε_pos, ε_neg) combinations found.")

    # Step 2: Generate data once and compute LP coefficients
    data_train = generate_data_pre(n, **data_params)
    data_train['R'] = (data_train.mu0 >= 0.5).astype(float)

    data_val = generate_data_pre(n, **data_params)
    data_val['R'] = (data_val.mu0 >= 0.5).astype(float)

    coefs_obj = risk_coefs(data_train, 'A', 'R', 'mu0')
    coefs_pos, coefs_neg = fairness_coefs(data_train, 'A', 'R', 'mu0')

    # Step 3: Define function for computing metrics for one (ε_pos, ε_neg)
    def evaluate_combo(eps_pos, eps_neg):
        theta = optimize(coefs_obj, coefs_pos, coefs_neg, eps_pos, eps_neg)
        df = metrics_post(theta, data_val, outcome='mu0', ci=None)
        df['epsilon_pos'] = eps_pos
        df['epsilon_neg'] = eps_neg
        return df

    # Step 4: Parallelize
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_combo)(eps_pos, eps_neg) for eps_pos, eps_neg in valid_combos
    )

    return pd.concat(results, ignore_index=True)


def add_reference_values(metrics_df, reference_values_df, value_col='value', new_col='reference_value'):
    """
    Merge metrics for a given theta-hat with corresponding reference metric values.

    In general, the reference metrics will be either (1) the values of the
    optimal fair predictor (for task 1) or the "true" values from the oracle (for task 2).

    Args:
        metrics_df (pd.DataFrame): Long-format DataFrame with simulation results.
            Must include a 'metric' column.
        reference_values_df (pd.DataFrame): DataFrame containing metrics for the
            optimal fair predictor, containing a 'metric' column and a column 
            (default 'value') to merge.
        value_col (str): Name of the column in `optimal_values_df` containing the
            values to be added to `metrics_df`.
        new_col (str): Name of the column to be added to `metrics_df` after the merge.

    Returns:
        pd.DataFrame: `metrics_df` with a new column `new_col` representing true values.

    Raises:
        ValueError: If required columns are missing.
    """
    required_cols = {'metric', value_col}
    if not required_cols.issubset(reference_values_df.columns):
        raise ValueError(f"`reference_values_df` must contain columns: {required_cols}")

    if 'metric' not in metrics_df.columns:
        raise ValueError("`metrics_df` must contain a 'metric' column for merging.")

    merged = metrics_df.merge(
        reference_values_df[['metric', value_col]].rename(columns={value_col: new_col}),
        on='metric',
        how='left'
    )

    return merged
