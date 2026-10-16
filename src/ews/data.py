"""Loading the weekly panel and splitting it by customer."""
import numpy as np
import pandas as pd

from . import config as C


def load_panel(path=C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[C.WEEK] = pd.to_datetime(df[C.WEEK])
    df = df.sort_values([C.ID, C.WEEK]).reset_index(drop=True)
    df[C.WEEK_IDX] = df.groupby(C.ID).cumcount()
    for c in C.CAT_COLS:
        df[c] = df[c].fillna("none").astype(str)
    add_emi_schedule(df)
    return df


def add_emi_schedule(df: pd.DataFrame) -> None:
    """Static loan-contract field: weeks until the next EMI due date (0 = due this week).

    Every loan in the panel is on a strict 4-week EMI cycle; the lender knows the
    due date from the contract on day one, so this is not future information.
    """
    first_due = (df[C.WEEK_IDX].where(df["emi_due_this_week"] == 1)
                 .groupby(df[C.ID]).transform("min").fillna(0).astype(int))
    df["emi_offset"] = first_due % 4
    df["weeks_to_emi_due"] = (df["emi_offset"] - df[C.WEEK_IDX]) % 4


def split_customers(df: pd.DataFrame, seed: int = C.SEED) -> dict:
    """Customer-level split into train / calib / test.

    Splitting by customer (not by row) matters: rows from the same person are
    highly correlated, so a random row split leaks a customer's future weeks
    into training and inflates every metric.
    """
    customers = np.array(sorted(df[C.ID].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(customers)
    n = len(customers)
    n_test = int(n * C.TEST_FRAC)
    n_cal = int(n * C.CALIB_FRAC)
    return {
        "test": set(customers[:n_test]),
        "calib": set(customers[n_test:n_test + n_cal]),
        "train": set(customers[n_test + n_cal:]),
    }


def customer_folds(customers, n_folds: int = C.N_FOLDS, seed: int = C.SEED):
    """Deterministic customer-grouped K folds."""
    customers = np.array(sorted(customers))
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(customers)
    return [set(f) for f in np.array_split(customers, n_folds)]
