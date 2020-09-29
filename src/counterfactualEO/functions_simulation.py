import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import confusion_matrix, classification_report
from .functions_estimation import (
    optimize, risk_coefs, fairness_coefs
)

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

    data = np.concatenate([AX, D, Y], axis=1)
    data = pd.DataFrame(data, columns=['A', 'X1', 'X2', 'X3', 'X4', 'D', 'Y'])

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
    """
    Solve the fairness-constrained linear program using true nuisance parameters.

    Generates a synthetic dataset, computes the objective and fairness constraint
    coefficients using true nuisance values (mu0), and solves for the optimal
    theta vector under fairness constraints.

    Args:
        data_params (dict): Parameters passed to the data generation function.
        epsilon_pos (float): Maximum allowed violation in FPR constraint.
        epsilon_neg (float): Maximum allowed violation in FNR constraint.
        n (int): Number of samples to generate for optimization.

    Returns:
        dict: Dictionary containing:
            - 'theta': Optimal decision vector.
            - 'obj': Objective coefficients (risk).
            - 'pos': Fairness constraint coefficients (FPR).
            - 'neg': Fairness constraint coefficients (FNR).
    """

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
                    noise_coef, trunc_pi=0.975, verbose=True):
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

    n_col = pd.Series([nn for nn in n_arr for k in range(mc_reps)] * 3)
    id_col = pd.Series(
        [mm for mm in ['obj', 'pos', 'neg'] for k in range(n_rows)])
    dist_col = pd.Series(np.concatenate([dist_obj, dist_pos, dist_neg]))

    comb = pd.DataFrame(np.concatenate([obj_arr, pos_arr, neg_arr], axis=0))
    out = pd.concat([n_col, id_col, comb, dist_col], axis=1)
    out.columns = ['n', 'id', 'comp0', 'comp1', 'comp2', 'comp3', 'L2']

    return out


def add_noise_expit(n_arr, mc_reps, data_params, obj_true, pos_true, neg_true,
                    trunc_pi=0.975, verbose=True):
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
        data_params (dict): Parameters for data generation.
        obj_true, pos_true, neg_true (np.ndarray): Ground truth LP coefficients.
        trunc_pi (float): Truncation value for pi.
        verbose (bool): Whether to print progress updates.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (Summary dataframe of results, 
        final dataset used).
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

    n_col = pd.Series([nn for nn in n_arr for k in range(mc_reps)] * 3)
    id_col = pd.Series(
        [mm for mm in ['obj', 'pos', 'neg'] for k in range(n_rows)])
    dist_col = pd.Series(np.concatenate([dist_obj, dist_pos, dist_neg]))

    comb = pd.DataFrame(np.concatenate([obj_arr, pos_arr, neg_arr], axis=0))
    out = pd.concat([n_col, id_col, comb, dist_col], axis=1)
    out.columns = ['n', 'id', 'comp0', 'comp1', 'comp2', 'comp3', 'L2']

    return out, data_train


def sim_theta(n, mc_reps, noise_coef, data_params, epsilon_pos, epsilon_neg,
              A='A', R='R', outcome='phihat', trunc_pi=0.975, verbose=False):
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
            phihat yields doubly robust estimators for the linear program, while
            muhat0 yields plugin estimators.
        trunc_pi (float): Maximum value for pi.
        verbose (bool): Whether to print progress.

    Returns:
        dict: Dictionary with sample size and theta estimates across repetitions.
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
    """
    Run a single experiment evaluating a fixed predictor under sampling variability.

    This function simulates the evaluation of a fixed predictor indexed by 
    `theta` by generating `mc_reps` datasets of size `n` using the provided data 
    generation process. It adds noise to the nuisance parameters and computes 
    fairness and risk metrics for each repetition, returning metric estimates 
    with confidence intervals.

    `Task 2` refers to the task of estimating counterfactual fairness and
    accuracy values for a fixed predictor, whereas `Task 1` refers to the task
    of estimating a counterfactually fair predictor.

    Args:
        theta (np.ndarray): Fixed predictor rule to evaluate (4-vector).
        noise_coef (float): Magnitude of noise added to nuisance parameters (phihat).
        n (int): Sample size per simulation repetition.
        mc_reps (int): Number of Monte Carlo repetitions.
        data_params (dict): Parameters used to generate the synthetic dataset.
        outcome (str): Column name for outcome to use ('phihat' or 'muhat0').
        trunc_pi (float): Upper bound for clipping estimated propensity scores.
        ci_scale (str): Scale for confidence intervals ('logit' or 'expit').
        verbose (bool): Whether to print progress updates.

    Returns:
        pd.DataFrame: DataFrame containing one row per metric per repetition, 
            with columns: ['mc_iter', 'metric', 'value', 'ci_lower', 'ci_upper'].
    """
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
    """
    Evaluate a fixed predictor across multiple sample sizes under repeated sampling.

    This is a wrapper around `_sim_task2` that runs it for each sample size in 
    `n_arr`, returning a combined DataFrame with all results. It is used to 
    study how estimation accuracy varies with dataset size for a fixed decision 
    rule.

    `Task 2` refers to the task of estimating counterfactual fairness and
    accuracy values for a fixed predictor, whereas `Task 1` refers to the task
    of estimating a counterfactually fair predictor.

    Args:
        theta (np.ndarray): Fixed decision vector (e.g., from an LP solution).
        noise_coef (float): Magnitude of nuisance parameter noise.
        n_arr (List[int]): List of sample sizes to evaluate.
        mc_reps (int): Number of Monte Carlo repetitions per sample size.
        data_params (dict): Parameters for generating synthetic data.
        trunc_pi (float): Truncation threshold for propensity scores.
        outcome (str): Outcome column used for evaluation ('phihat' or 'muhat0').
        ci_scale (str): Confidence interval scale ('logit' or 'expit').
        verbose (bool): Whether to display progress during simulation.

    Returns:
        pd.DataFrame: Combined long-format metric estimates across all sample 
            sizes, with columns: ['n', 'mc_iter', 'metric', 'value', 'ci_lower', 'ci_upper'].
    """
    metrics_list = [_sim_task2(theta, noise_coef, n, mc_reps,
                               data_params, outcome=outcome,
                               ci_scale=ci_scale, verbose=verbose) for n in n_arr]
    metrics_est = pd.concat(metrics_list, keys=n_arr)
    metrics_est = metrics_est.reset_index().drop(columns='level_1')
    metrics_est.columns = ['n', 'mc_iter', 'metric', 'value', 'ci_lower', 'ci_upper']

    return metrics_est


def simulate_true(n, data_params, epsilon_pos, epsilon_neg):
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
    evals = metrics(theta, data_val, A='A', R='R', outcome='Y0')

    return {
        'theta': theta,
        'risk_coefs': risk_coefs_,
        'fairness_coefs_pos': fair_pos,
        'fairness_coefs_neg': fair_neg,
        'metrics': evals
    }

# def simulate_true(n, data_params):
#     """Get the 'true' best fair predictor, given access to true values of the
#     nuisance parameters."""
#     ## Generate data
#     data_opt = generate_data_post(n, **data_params)

#     ## Get empirical coefficients for loss and fairness constraints
#     obj = risk_coefs(test, 'A', 'R', 'mu0')
#     fair_pos, fair_neg = fairness_coefs(test, 'A', 'R', 'Y0')

#     ## Get best derived predictor
#     theta = optimize(loss_true, coefs_pos, coefs_neg, epsilon_pos, epsilon_neg)

#     ## Get metrics of best derived predictor
#     data_val = generate_data_post(n, **data_params)
#     metrics = metrics(theta, data_val, 'A', 'R', 'Y0')
#     out = {'theta': theta, 'risk_coefs': obj,
#            'fairness_coefs_pos': fair_pos, 'fairness_coefs_neg': fair_neg,
#            'metrics': metrics}

#     return out
