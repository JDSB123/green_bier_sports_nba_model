import os
import tempfile

import pandas as pd

from src.modeling.models import SpreadsModel


def make_dummy_data(n=100):
    import numpy as np

    rng = np.random.default_rng(42)
    home_ppg = rng.normal(110, 5, size=n)
    away_ppg = rng.normal(108, 5, size=n)
    predicted_margin = home_ppg - away_ppg
    X = pd.DataFrame(
        {
            "home_ppg": home_ppg,
            "away_ppg": away_ppg,
            "predicted_margin": predicted_margin,
        }
    )
    # target: 1 if predicted_margin > 0
    y = (predicted_margin > 0).astype(int)
    return X, y


def test_save_and_load_model():
    X, y = make_dummy_data(50)
    model = SpreadsModel(
        model_type="logistic", feature_columns=["home_ppg", "away_ppg", "predicted_margin"]
    )
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "spreads_test.joblib")
        model.save(path)

        # load into a fresh instance
        m2 = SpreadsModel()
        m2.load(path)

        preds = m2.predict(X)
        assert len(preds) == len(X)
        # probabilities
        probas = m2.predict_proba(X)
        assert probas.shape[0] == len(X)
