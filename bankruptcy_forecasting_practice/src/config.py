from pathlib import Path
import yaml


def load_config(config_path: str = "config/params.yaml") -> dict:
    """Load YAML parameters and return them as a dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required = ["actuals_till", "forecast_start", "forecast_end", "model", "paths"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")

    return config
