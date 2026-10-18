import pandas as pd

from src.features import NUMERIC_FEATURES


def build_forecast_base(model_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Project AR and delinquency assumptions across the requested forecast horizon."""
    assumptions = config["forecast_assumptions"]
    forecast_dates = pd.date_range(
        start=config["forecast_start"],
        end=config["forecast_end"],
        freq="ME",
    )

    base_cols = [
        "customer_id",
        "industry",
        "ar_balance",
        "dq_ar_balance",
        "dq_ratio",
        "leverage_ratio",
        "current_ratio",
        "interest_coverage_ratio",
        "cash_ratio",
    ]
    latest = model_df[base_cols].copy()

    forecasts = []
    for month_number, forecast_date in enumerate(forecast_dates, start=1):
        monthly = latest.copy()
        monthly["forecast_date"] = forecast_date

        monthly["ar_balance"] = (
            monthly["ar_balance"]
            * (1 + assumptions["monthly_ar_growth_rate"]) ** month_number
        )

        monthly["dq_ratio"] = (
            monthly["dq_ratio"]
            + assumptions["monthly_dq_ratio_deterioration"] * month_number
        ).clip(upper=assumptions["max_dq_ratio"])

        monthly["dq_ar_balance"] = monthly["ar_balance"] * monthly["dq_ratio"]
        forecasts.append(monthly)

    return pd.concat(forecasts, ignore_index=True)


def score_forecast(forecast_df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """Score each customer-month with a forecast probability of bankruptcy."""
    output = forecast_df.copy()
    output["bankruptcy_probability"] = pipeline.predict_proba(
        output[NUMERIC_FEATURES]
    )[:, 1]

    threshold = config["model"]["bankruptcy_probability_threshold"]
    output["high_risk_flag"] = (
        output["bankruptcy_probability"] >= threshold
    ).astype(int)

    return output


def create_monthly_summary(scored_forecast: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customer-level results into a portfolio-level monthly view."""
    summary = (
        scored_forecast
        .groupby("forecast_date", as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            total_ar=("ar_balance", "sum"),
            total_dq_ar=("dq_ar_balance", "sum"),
            avg_bankruptcy_probability=("bankruptcy_probability", "mean"),
            high_risk_customers=("high_risk_flag", "sum"),
        )
    )

    summary["portfolio_dq_ratio"] = summary["total_dq_ar"] / summary["total_ar"]
    return summary
