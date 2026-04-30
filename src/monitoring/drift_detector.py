import json
import os
import requests
import logging
from datetime import datetime
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
REFERENCE_DATA_PATH = "data/processed/reference_data.csv"
CURRENT_DATA_PATH = (
    "data/processed/production_logs.csv"  # In reality, fetched from your DB
)
REPORT_HTML_PATH = "dashboard/reports/evidently_report.html"
REPORT_JSON_PATH = "dashboard/reports/evidently_report.json"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def generate_simulated_data():
    """Simulates loading reference and current production data for this example."""
    logging.info("Loading dataset for monitoring simulation...")

    file_path = "data/raw/application_train.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path).head(10000)
    else:
        logging.warning("Raw data not found. Generating synthetic data for CI/CD...")
        import numpy as np
        np.random.seed(42)
        n_samples = 10000
        # Create 10 features to keep drift share below 0.5 when 2 features drift
        data = {"TARGET": np.random.randint(0, 2, n_samples)}
        for i in range(1, 9):
            data[f"FEATURE_{i}"] = np.random.normal(0, 1, n_samples)
        data["AMT_INCOME_TOTAL"] = np.random.normal(150000, 50000, n_samples)
        data["DAYS_EMPLOYED"] = np.random.normal(-2000, 1000, n_samples)
        df = pd.DataFrame(data)

    # Split into reference (past) and current (recent production)
    reference_df = df.iloc[:5000].copy()
    current_df = df.iloc[5000:10000].copy()

    # Simulate a sudden data drift in the production environment (e.g., economic crash)
    current_df["AMT_INCOME_TOTAL"] = current_df["AMT_INCOME_TOTAL"] * 0.5
    current_df["DAYS_EMPLOYED"] = current_df["DAYS_EMPLOYED"] * 1.5

    # Simulate predictions
    reference_df["prediction"] = reference_df["TARGET"]
    current_df["prediction"] = current_df["TARGET"]

    return reference_df, current_df


def run_evidently(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    """Runs Evidently AI drift detection and saves reports."""
    logging.info("Running Evidently AI Data and Target Drift analysis...")

    os.makedirs("dashboard/reports", exist_ok=True)

    # Evidently 0.4+ DataDriftPreset includes dataset and column drift metrics
    drift_report = Report(metrics=[DataDriftPreset()])

    snapshot = drift_report.run(reference_data=reference_df, current_data=current_df)

    # Save artifacts for Streamlit
    snapshot.save_html(REPORT_HTML_PATH)
    snapshot.save_json(REPORT_JSON_PATH)
    logging.info(f"Reports saved to {REPORT_HTML_PATH} and {REPORT_JSON_PATH}")


def check_thresholds_and_alert():
    """Parses the JSON report and sends a Discord alert if drift is detected."""
    logging.info("Analyzing drift results...")

    with open(REPORT_JSON_PATH, "r") as f:
        results = json.load(f)

    # Extract data drift metric from Evidently 0.7.x JSON format
    drifted_columns = 0
    total_columns = 1
    dataset_drift = False

    for m in results.get("metrics", []):
        name = m.get("metric_name", "")
        if name.startswith("DriftedColumnsCount"):
            val = m.get("value", {})
            drifted_columns = int(val.get("count", 0))
            share = val.get("share", 0)
            if share > 0:
                total_columns = int(round(drifted_columns / share))
            dataset_drift = share >= 0.5  # Adjust threshold if needed
            break

    logging.info(
        f"Drift Detected: {dataset_drift}. Drifted Columns: {drifted_columns}/{total_columns}"
    )

    if dataset_drift and DISCORD_WEBHOOK_URL:
        logging.warning("CRITICAL: Data drift detected! Sending Discord alert...")

        message = {
            "content": f"🚨 **MLOps Alert: Data Drift Detected!** 🚨\n\n"
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Model:** CreditRiskModel (Production)\n"
            f"**Details:** {drifted_columns} out of {total_columns} features have drifted significantly.\n"
            f"**Action Required:** Check the Streamlit Monitoring Dashboard immediately to view the Evidently AI report and consider retraining."
        }

        response = requests.post(DISCORD_WEBHOOK_URL, json=message)
        if response.status_code == 204:
            logging.info("Discord alert sent successfully.")
        else:
            logging.error(f"Failed to send Discord alert: {response.status_code}")


def main():
    ref_df, cur_df = generate_simulated_data()
    run_evidently(ref_df, cur_df)
    check_thresholds_and_alert()


if __name__ == "__main__":
    main()
