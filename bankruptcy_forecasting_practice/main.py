from src.config import load_config
from src.data_loader import load_source_data
from src.validation import validate_inputs
from src.features import build_training_dataset
from src.model import train_bankruptcy_model
from src.forecast import build_forecast_base, score_forecast, create_monthly_summary
from src.reporting import save_outputs


def main() -> None:
    print("1. Loading parameters...")
    config = load_config()

    print("2. Loading CSV and database inputs...")
    ar, dq_ar, ratios, labels = load_source_data(config)

    print("3. Validating inputs...")
    validate_inputs(ar, dq_ar, ratios, labels, config)

    print("4. Building modelling dataset...")
    model_df = build_training_dataset(
        ar=ar,
        dq_ar=dq_ar,
        ratios=ratios,
        labels=labels,
        actuals_till=config["actuals_till"],
    )

    print("5. Training bankruptcy model...")
    model_result = train_bankruptcy_model(model_df, config)

    print("6. Building forecast horizon...")
    forecast_base = build_forecast_base(model_df, config)

    print("7. Scoring customer bankruptcy probabilities...")
    scored_forecast = score_forecast(
        forecast_base,
        model_result.pipeline,
        config,
    )

    print("8. Creating portfolio summary...")
    monthly_summary = create_monthly_summary(scored_forecast)

    print("9. Saving outputs...")
    save_outputs(
        scored_forecast=scored_forecast,
        monthly_summary=monthly_summary,
        metrics=model_result.metrics,
        output_folder=config["paths"]["output_folder"],
    )

    print("\nPipeline completed successfully.")
    print(f"ROC-AUC: {model_result.metrics['roc_auc']}")
    print("Outputs written to the output/ folder.")


if __name__ == "__main__":
    main()
