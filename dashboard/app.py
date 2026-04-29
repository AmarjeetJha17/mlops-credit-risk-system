import streamlit as st
import streamlit.components.v1 as components
import json
import os
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="MLOps Monitoring", page_icon="📊", layout="wide")

REPORT_HTML_PATH = "dashboard/reports/evidently_report.html"
REPORT_JSON_PATH = "dashboard/reports/evidently_report.json"

st.title("📊 Credit Risk Model: Production Monitoring")
st.markdown("Automated data drift and performance tracking using Evidently AI.")

if not os.path.exists(REPORT_HTML_PATH):
    st.warning("No drift report found. Please run the drift detector pipeline first.")
else:
    # Read the JSON for high-level KPIs
    with open(REPORT_JSON_PATH, 'r') as f:
        results = json.load(f)
        
    drift_metrics = results["metrics"][0]["result"]
    is_drifted = drift_metrics["dataset_drift"]
    drift_share = drift_metrics["share_of_drifted_columns"]
    
    # Dashboard KPI Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_color = "red" if is_drifted else "green"
        status_text = "DRIFT DETECTED" if is_drifted else "HEALTHY"
        st.markdown(f"### System Status\n<h2 style='color: {status_color};'>{status_text}</h2>", unsafe_allow_html=True)
        
    with col2:
        st.metric(label="Drifted Features (%)", value=f"{drift_share * 100:.1f}%")
        
    with col3:
        st.metric(label="Last Updated", value=datetime.fromtimestamp(os.path.getmtime(REPORT_JSON_PATH)).strftime('%Y-%m-%d %H:%M'))

    st.markdown("---")
    st.subheader("Deep Dive: Evidently AI Interactive Report")
    st.markdown("Review the exact distributions that caused the drift alerts below.")
    
    # Render the Evidently HTML report inside Streamlit
    with open(REPORT_HTML_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    components.html(html_content, height=1000, scrolling=True)