# Bankruptcy Forecasting Practice Project

A small, modular Data Science project designed for practising:

- Python project structure in VS Code
- Git add / commit / branch / merge
- GitHub push / pull
- Parameter-driven pipelines
- CSV inputs
- Database reads using SQL
- Feature engineering
- Model training
- Forecast generation
- Output persistence

## Business use case

The project estimates the probability that a bank customer/counterparty becomes bankrupt within the next 12 months.

Illustrative drivers:

- Accounts Receivable (AR)
- Delinquent / DQ Accounts Receivable
- DQ-to-AR ratio
- Leverage ratio
- Current ratio
- Interest coverage ratio
- Cash ratio

`ar_actuals.csv`, customer ratios and labels are treated as uploaded files.

DQ AR is read from a SQLite database to simulate a database extract.

## Folder structure

```text
bankruptcy_forecasting_practice/
├── config/
│   └── params.yaml
├── database/
│   └── bank.db
├── input/
│   ├── ar_actuals.csv
│   ├── customer_ratios.csv
│   └── historical_bankruptcy_labels.csv
├── output/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_io.py
│   ├── data_loader.py
│   ├── features.py
│   ├── forecast.py
│   ├── model.py
│   ├── reporting.py
│   └── validation.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Run locally

### 1. Open the folder in VS Code

### 2. Create a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python main.py
```

Expected generated files:

```text
output/customer_bankruptcy_forecast.csv
output/portfolio_monthly_summary.csv
output/model_metrics.json
```

## Suggested Git practice

Initialise the repository:

```bash
git init
git add .
git commit -m "Initial modular bankruptcy forecasting pipeline"
```

Create a feature branch:

```bash
git switch -c feature/add-risk-bands
```

Try adding a `risk_band` field to `src/forecast.py`, then:

```bash
git status
git diff
git add src/forecast.py
git commit -m "Add customer bankruptcy risk bands"
```

Return to main:

```bash
git switch main
git merge feature/add-risk-bands
```

After creating an empty GitHub repository:

```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git branch -M main
git push -u origin main
```

Later practise:

```bash
git pull
git push
git log --oneline --graph --all
```

## Important modelling note

This is intentionally a learning project. The data is synthetic and the forecast assumptions are illustrative. A real credit-risk/bankruptcy model would require stronger target definition, point-in-time features, leakage controls, temporal validation, class-imbalance treatment, calibration, explainability, governance and monitoring.
