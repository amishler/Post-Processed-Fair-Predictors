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
