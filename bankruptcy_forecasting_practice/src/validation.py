import pandas as pd


def validate_inputs(
    ar: pd.DataFrame,
    dq_ar: pd.DataFrame,
    ratios: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict,
) -> None:
    """Run basic data-quality checks before modelling."""
    actuals_till = pd.Timestamp(config["actuals_till"])

    required_ar = {"customer_id", "as_of_date", "ar_balance"}
    required_dq = {"customer_id", "as_of_date", "dq_ar_balance"}
    required_ratios = {
        "customer_id",
        "leverage_ratio",
        "current_ratio",
        "interest_coverage_ratio",
        "cash_ratio",
    }
    required_labels = {"customer_id", "bankrupt_within_12m"}

    checks = [
        ("AR", required_ar, set(ar.columns)),
        ("DQ AR", required_dq, set(dq_ar.columns)),
        ("Ratios", required_ratios, set(ratios.columns)),
        ("Labels", required_labels, set(labels.columns)),
    ]

    for dataset_name, required, actual in checks:
        missing = required - actual
        if missing:
            raise ValueError(f"{dataset_name} is missing columns: {sorted(missing)}")

    if ar["customer_id"].isna().any() or dq_ar["customer_id"].isna().any():
        raise ValueError("customer_id contains null values.")

    if (ar["ar_balance"] < 0).any() or (dq_ar["dq_ar_balance"] < 0).any():
        raise ValueError("AR and DQ AR balances must be non-negative.")

    if ar["as_of_date"].max() < actuals_till:
        raise ValueError(
            f"AR data only runs through {ar['as_of_date'].max().date()}, "
            f"but actuals_till is {actuals_till.date()}."
        )
