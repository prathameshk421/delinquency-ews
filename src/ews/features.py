"""Feature construction for both model families.

Everything here is *causal*: a row for week t only ever uses weeks <= t of the
same customer, so the same code is valid in training and in live scoring.
"""
import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------- tabular ---
def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag / rolling / trend features per customer (past-only)."""
    df = df.sort_values([C.ID, C.WEEK_IDX]).copy()
    g = df.groupby(C.ID, sort=False)
    new = {}
    for c in C.TREND_COLS:
        s = g[c]
        new[f"{c}__d1"] = s.diff().fillna(0.0)
        new[f"{c}__rm3"] = s.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        new[f"{c}__ewm"] = s.ewm(alpha=0.5, adjust=False).mean().reset_index(level=0, drop=True)
        new[f"{c}__vs_first"] = df[c] - s.transform("first")
        new[f"{c}__max_sofar"] = s.cummax()
    new["emi_bounced__cum"] = g["emi_bounced_flag"].cumsum()
    new["shock__cum"] = g["external_shock_flag"].cumsum()
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def tabular_columns(with_trends: bool = True, with_schedule: bool = True) -> list:
    base = C.STATIC_NUM_COLS + C.CAT_COLS + C.BEHAVIOURAL + [C.WEEK_IDX, "weeks_to_emi_due"]
    if not with_schedule:
        base = [c for c in base if c != "weeks_to_emi_due"]
    if not with_trends:
        return base
    trend = [f"{c}__{s}" for c in C.TREND_COLS for s in ("d1", "rm3", "ewm", "vs_first", "max_sofar")]
    return base + trend + ["emi_bounced__cum", "shock__cum"]


class CategoryCodec:
    """Fixed category levels so train-time and serve-time encodings agree."""

    def __init__(self):
        self.levels = {}

    def fit(self, df):
        self.levels = {c: sorted(df[c].astype(str).unique().tolist()) for c in C.CAT_COLS}
        return self

    def transform(self, df):
        df = df.copy()
        for c in C.CAT_COLS:
            df[c] = pd.Categorical(df[c].astype(str), categories=self.levels[c])
        return df


# ------------------------------------------------------------- sequential ---
class SeqScaler:
    """Signed-log for money columns, then standardisation (fit on train rows)."""

    def __init__(self, cols):
        self.cols = list(cols)
        self.log_mask = np.array([c in C.SIGNED_LOG_COLS for c in self.cols])

    def _pre(self, X):
        X = X.astype(np.float32).copy()
        X[:, self.log_mask] = np.sign(X[:, self.log_mask]) * np.log1p(np.abs(X[:, self.log_mask]))
        return X

    def fit(self, df):
        X = self._pre(df[self.cols].to_numpy())
        self.mean_ = X.mean(0)
        self.std_ = X.std(0) + 1e-6
        return self

    def transform(self, df):
        return (self._pre(df[self.cols].to_numpy()) - self.mean_) / self.std_


def build_windows(df: pd.DataFrame, scaler: SeqScaler, seq_len: int = C.SEQ_LEN):
    """One left-padded window per (customer, week) row, ending at that week.

    Returns X [N, seq_len, F], mask [N, seq_len] (1 = real week), y [N] and the
    row index each window belongs to (so predictions line up with `df`).
    The window for week t contains weeks max(0, t-seq_len+1) .. t only, which is
    what makes the sequence model causal.
    """
    df = df.sort_values([C.ID, C.WEEK_IDX])
    Z = scaler.transform(df)
    F = Z.shape[1]
    N = len(df)
    X = np.zeros((N, seq_len, F), dtype=np.float32)
    M = np.zeros((N, seq_len), dtype=np.float32)
    ids = df[C.ID].to_numpy()
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], N]
    for s, e in zip(starts, ends):
        for t in range(s, e):
            lo = max(s, t - seq_len + 1)
            L = t - lo + 1
            X[t, seq_len - L:] = Z[lo:t + 1]
            M[t, seq_len - L:] = 1.0
    y = df[C.TARGET].to_numpy().astype(np.float32) if C.TARGET in df else None
    return X, M, y, df.index.to_numpy()
