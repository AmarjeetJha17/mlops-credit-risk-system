import pandas as pd
import json
import os
import requests
import logging
from datetime import datetime
from evidently.report.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

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

    # We will use the raw Kaggle dataset to simulate this
    # In a real system, 'current_df' comes from the FastAPI request logs
    df = pd.read_csv("data/raw/application_train.csv").head(10000)

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

    drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])

    drift_report.run(reference_data=reference_df, current_data=current_df)

    # Save artifacts for Streamlit
    drift_report.save_html(REPORT_HTML_PATH)
    drift_report.save_json(REPORT_JSON_PATH)
    logging.info(f"Reports saved to {REPORT_HTML_PATH} and {REPORT_JSON_PATH}")


def check_thresholds_and_alert():
    """Parses the JSON report and sends a Discord alert if drift is detected."""
    logging.info("Analyzing drift results...")

    with open(REPORT_JSON_PATH, "r") as f:
        results = json.load(f)

    # Extract data drift metric
    drift_metrics = results["metrics"][0]["result"]
    dataset_drift = drift_metrics["dataset_drift"]
    drifted_columns = drift_metrics["number_of_drifted_columns"]
    total_columns = drift_metrics["number_of_columns"]

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
