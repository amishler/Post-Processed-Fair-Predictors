# tests/conftest.py
from pathlib import Path
import pytest
import pandas as pd
import pickle

@pytest.fixture(scope="session")
def real_data():
    base_path = Path(__file__).parent
    df = pd.read_csv(base_path / "test_data.csv")
    with open(base_path / "test_values1.pickle", "rb") as f:
        test_values = pickle.load(f)
    return df, test_values
