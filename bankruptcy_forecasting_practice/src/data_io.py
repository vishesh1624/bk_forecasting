from pathlib import Path
import sqlite3
import pandas as pd


def read_csv(path: str, parse_dates=None) -> pd.DataFrame:
    """Read an uploaded CSV file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    return pd.read_csv(file_path, parse_dates=parse_dates)


def read_dq_ar_from_database(db_path: str, actuals_till: str) -> pd.DataFrame:
    """Simulate pulling DQ AR actuals from a database using SQLite."""
    query = """
        SELECT customer_id, as_of_date, dq_ar_balance
        FROM dq_ar_actuals
        WHERE date(as_of_date) <= date(?)
    """

    with sqlite3.connect(db_path) as connection:
        df = pd.read_sql_query(
            query,
            connection,
            params=[actuals_till],
            parse_dates=["as_of_date"],
        )
    return df


def write_csv(df: pd.DataFrame, path: str) -> None:
    """Write dataframe to CSV, creating the parent folder if required."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
