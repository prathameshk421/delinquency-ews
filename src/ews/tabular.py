"""LightGBM tabular model helpers."""
import lightgbm as lgb
import numpy as np

from . import config as C


def fit_lgb(Xtr, ytr, Xes, yes, params=None):
    """Fit with early stopping on a held-aside *training* subset (never on test/calib)."""
    p = {**C.LGB_PARAMS, **(params or {}), "random_state": C.SEED}
    model = lgb.LGBMClassifier(**p)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xes, yes)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def lgb_contribs(booster, X):
    """Exact TreeSHAP values from LightGBM itself (last column = bias), in log-odds."""
    return booster.predict(X, pred_contrib=True)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
