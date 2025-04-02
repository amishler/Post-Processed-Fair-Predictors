import pytest

import numpy as np
import pandas as pd

from counterfactualEO.functions_simulation import (
    generate_data_pre,
    generate_data_post,
    generate_data_post_noisy,
    dist_to_ref,
    add_noise_logit,
    add_noise_expit,
    sim_task2,
    _sim_task2,
    sim_theta,
    simulate_true
)


def test_generate_data_pre_shapes():
    df = generate_data_pre(
        n=100, prob_A=0.3, 
        beta_X=np.ones(1), 
        beta_D=np.ones(5), 
        beta_Y0=np.ones(5), 
        beta_Y1=np.ones(5)
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 7


def test_generate_data_post_shapes(mock_model):
    df = generate_data_post(
        n=100, prob_A=0.3, 
        beta_X=np.ones(1), 
        beta_D=np.ones(6), 
        beta_Y0=np.ones(5), 
        beta_Y1=np.ones(5), 
        model_R=mock_model
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 13


def test_generate_data_post_noisy_shapes(mock_model):
    df = generate_data_post_noisy(
        n=100, noise_coef=0.5, prob_A=0.3, 
        beta_X=np.ones(1), 
        beta_D=np.ones(6), 
        beta_Y0=np.ones(5), 
        beta_Y1=np.ones(5), 
        model_R=mock_model
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 16


def test_dist_to_ref():
    arr = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    vec = np.array([1, 2, 3, 4])
    dists = dist_to_ref(arr, vec)
    assert np.allclose(dists, [0, np.linalg.norm([3, 1, -1, -3])])


def test_sim_theta_output(mock_data_params):
    out = sim_theta(n=50, mc_reps=3, noise_coef=0.1, 
                       data_params=mock_data_params,
                       epsilon_pos=0.1, epsilon_neg=0.1)
    assert 'n' in out and 'theta_arr' in out
    assert out['theta_arr'].shape == (3, 4)


def test_simulate_true_output(mock_data_params):
    out = simulate_true(n=100, data_params=mock_data_params, 
                           epsilon_pos=0.1, epsilon_neg=0.1)
    assert isinstance(out, dict)
    assert all(k in out for k in ['theta', 'risk_coefs', 'fairness_coefs_pos', 
                                  'fairness_coefs_neg', 'metrics'])


@pytest.mark.parametrize("n_arr, mc_reps", [([50], 3)])
def test_add_noise_logit_outputs_shape(mock_data_params, mock_true_coefs, n_arr, mc_reps):
    df = add_noise_logit(n_arr=n_arr,
                         mc_reps=mc_reps,
                         data_params=mock_data_params,
                         obj_true=mock_true_coefs["obj"],
                         pos_true=mock_true_coefs["pos"],
                         neg_true=mock_true_coefs["neg"],
                         noise_coef=0.1,
                         verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert "L2" in df.columns
    assert df.shape[0] == len(n_arr) * mc_reps * 3


@pytest.mark.parametrize("n_arr, mc_reps", [([50], 3)])
def test_add_noise_expit_outputs_shape(mock_data_params, mock_true_coefs, n_arr, mc_reps):
    df, _ = add_noise_expit(n_arr=n_arr,
                            mc_reps=mc_reps,
                            data_params=mock_data_params,
                            obj_true=mock_true_coefs["obj"],
                            pos_true=mock_true_coefs["pos"],
                            neg_true=mock_true_coefs["neg"],
                            trunc_pi=0.975,
                            verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert "L2" in df.columns
    assert df.shape[0] == len(n_arr) * mc_reps * 3


def test_sim_task2_format(mock_data_params, mock_theta):
    out = sim_task2(mock_theta, noise_coef=0.1, n_arr=[50], mc_reps=3, data_params=mock_data_params)
    assert isinstance(out, pd.DataFrame)
    assert set(["n", "mc_iter", "metric", "value", "ci_lower", "ci_upper"]).issubset(out.columns)


def test__sim_task2_format(mock_data_params, mock_theta):
    out = _sim_task2(mock_theta, noise_coef=0.1, n=50, mc_reps=3, data_params=mock_data_params)
    assert isinstance(out, pd.DataFrame)
    assert set(["mc_iter", "metric", "value", "ci_lower", "ci_upper"]).issubset(out.columns)
