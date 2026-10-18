import pandas as pd


NUMERIC_FEATURES = [
    "ar_balance",
    "dq_ar_balance",
    "dq_ratio",
    "leverage_ratio",
    "current_ratio",
    "interest_coverage_ratio",
    "cash_ratio",
]


def build_training_dataset(
    ar: pd.DataFrame,
    dq_ar: pd.DataFrame,
    ratios: pd.DataFrame,
    labels: pd.DataFrame,
    actuals_till: str,
) -> pd.DataFrame:
    """Create one modelling row per customer using the latest actual period."""
    actuals_till = pd.Timestamp(actuals_till)

    ar_latest = ar[ar["as_of_date"] == actuals_till].copy()
    dq_latest = dq_ar[dq_ar["as_of_date"] == actuals_till].copy()

    model_df = (
        ar_latest
        .merge(dq_latest, on=["customer_id", "as_of_date"], how="inner")
        .merge(ratios, on="customer_id", how="left")
        .merge(labels, on="customer_id", how="inner")
    )

    model_df["dq_ratio"] = (
        model_df["dq_ar_balance"] / model_df["ar_balance"].replace(0, pd.NA)
    ).fillna(0)

    return model_df
