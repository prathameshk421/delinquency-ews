"""History-aware stacking + calibrator selection.

The meta-learner sees each base model's log-odds plus an interaction with an
"early history" flag. In weeks 1-2 the GRU has almost nothing to read, so the
meta-learner can learn to lean on LightGBM there and on the GRU afterwards.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from .tabular import logit

EARLY_WEEKS = 2  # week_idx < 2  ->  fewer than 3 weeks of history
META_FEATURES = ["lgb_logit", "gru_logit", "lgb_logit x early", "gru_logit x early", "early"]


def meta_features(p_lgb, p_gru, week_idx):
    early = (np.asarray(week_idx) < EARLY_WEEKS).astype(float)
    a, b = logit(np.asarray(p_lgb)), logit(np.asarray(p_gru))
    return np.c_[a, b, a * early, b * early, early]


def fit_meta(p_lgb, p_gru, week_idx, y):
    return LogisticRegression(C=1.0, max_iter=1000).fit(meta_features(p_lgb, p_gru, week_idx), y)


class Calibrator:
    """Wraps either isotonic or Platt scaling behind one predict()."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, s, y):
        if self.kind == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)
        else:
            self.m = LogisticRegression().fit(logit(s).reshape(-1, 1), y)
        return self

    def predict(self, s):
        s = np.asarray(s, dtype=float)
        if self.kind == "isotonic":
            return self.m.predict(s)
        return self.m.predict_proba(logit(s).reshape(-1, 1))[:, 1]


def choose_calibrator(s, y, groups, n_folds=5, seed=0):
    """Pick isotonic vs Platt by customer-grouped CV Brier score on the calib split."""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    folds = np.array_split(ug, n_folds)
    scores = {}
    for kind in ("isotonic", "platt"):
        b = []
        for f in folds:
            te = np.isin(groups, f)
            cal = Calibrator(kind).fit(s[~te], y[~te])
            b.append(brier_score_loss(y[te], cal.predict(s[te])))
        scores[kind] = float(np.mean(b))
    best = min(scores, key=scores.get)
    return Calibrator(best).fit(s, y), scores
