from pathlib import Path
import json

from src.data_io import write_csv


def save_outputs(scored_forecast, monthly_summary, metrics, output_folder: str) -> None:
    """Persist model outputs and metrics."""
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)

    write_csv(scored_forecast, folder / "customer_bankruptcy_forecast.csv")
    write_csv(monthly_summary, folder / "portfolio_monthly_summary.csv")

    with (folder / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
