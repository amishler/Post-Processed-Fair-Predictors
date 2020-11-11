from pathlib import Path
import pytest

import numpy as np
import pandas as pd
import pickle


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


@pytest.fixture(scope="session")
def real_data():
    base_path = Path(__file__).parent  # this resolves to tests/
    df = pd.read_csv(base_path / "test_data.csv")
    with open(base_path / "test_values1.pickle", "rb") as f:
        test_values = pickle.load(f)
    return df, test_values


class MockRiskModel:
    """Simple mock model that returns deterministic predictions."""
    def predict(self, X):
        return np.random.binomial(1, 0.5, size=(X.shape[0],))


@pytest.fixture
def mock_model():
    """Returns a simple binary classifier mock."""
    return MockRiskModel()


@pytest.fixture
def mock_data_params(mock_model):
    """Returns synthetic parameter values for generating mock data."""
    return {
        'prob_A': 0.3,
        'beta_X': np.array([0.5, 0.2, -0.1, 0.3]),
        'beta_D': np.array([0.2, 0.1, -0.3, 0.2, -0.1, 0.4]),  # AXR: A + 4 Xs + R = 6 features
        'beta_Y0': np.array([0.3, 0.2, 0.1, -0.2, 0.1]),
        'beta_Y1': np.array([-0.1, 0.4, 0.2, 0.1, 0.2]),
        'model_R': mock_model,
    }


@pytest.fixture
def mock_theta():
    """Returns a simple valid theta vector (e.g. from optimization)."""
    return np.array([0.2, 0.3, 0.1, 0.4])


@pytest.fixture
def mock_true_coefs():
    """Returns synthetic 'true' LP coefficients for testing distance comparisons."""
    return {
        "obj": np.array([0.1, 0.2, -0.1, 0.0]),
        "pos": np.array([0.3, 0.2, -0.2, -0.1]),
        "neg": np.array([-0.1, -0.2, 0.2, 0.1]),
    }


@pytest.fixture
def mock_metrics_true():
    return pd.DataFrame({
        'metric': ['risk', 'risk_change'],
        'value': [0.2, 0.1]
    })


@pytest.fixture
def mock_metrics_est_with_n():
    return pd.DataFrame({
        'n': [50, 50],
        'metric': ['risk', 'risk_change'],
        'value': [0.21, 0.11],
        'ci_lower': [0.19, 0.09],
        'ci_upper': [0.23, 0.13]
    })
