import pandas as pd

from src.data_io import read_csv, read_dq_ar_from_database


def load_source_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load uploaded CSV sources and database-sourced DQ AR actuals."""
    paths = config["paths"]

    ar = read_csv(paths["ar_csv"], parse_dates=["as_of_date"])
    ratios = read_csv(paths["ratios_csv"])
    labels = read_csv(paths["labels_csv"])
    dq_ar = read_dq_ar_from_database(paths["sqlite_db"], config["actuals_till"])

    return ar, dq_ar, ratios, labels
