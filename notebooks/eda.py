import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Constants and Paths
RAW_DATA_PATH = "data/raw/application_train.csv"
FIG_DIR = "reports/figures"
STATS_DIR = "reports/stats"

# Ensure output directories exist
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

def load_data(path: str) -> pd.DataFrame:
    """Loads the dataset."""
    logging.info(f"Loading data from {path}...")
    df = pd.read_csv(path)
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
    return df

def analyze_target_distribution(df: pd.DataFrame):
    """Analyzes and plots class imbalance."""
    logging.info("Analyzing Target Distribution...")
    
    # Calculate stats
    target_counts = df['TARGET'].value_counts()
    target_pct = df['TARGET'].value_counts(normalize=True) * 100
    
    stats_df = pd.DataFrame({'Count': target_counts, 'Percentage': target_pct})
    stats_df.to_csv(f"{STATS_DIR}/target_distribution.csv")
    
    # Plot
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='TARGET', palette='viridis')
    plt.title('Target Distribution (0 = Repaid, 1 = Default)')
    plt.xlabel('Target')
    plt.ylabel('Number of Loans')
    
    # Add percentage text on bars
    for i, p in enumerate(target_pct):
        plt.text(i, target_counts[i] + 1000, f"{p:.2f}%", ha='center')
        
    plt.savefig(f"{FIG_DIR}/target_distribution.png", bbox_inches='tight')
    plt.close()

def analyze_missing_values(df: pd.DataFrame):
    """Calculates missing values and plots the top 20 worst columns."""
    logging.info("Analyzing Missing Values...")
    
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing_data, 'Missing_Percent': missing_pct})
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)
    
    # Save full stats
    missing_df.to_csv(f"{STATS_DIR}/missing_values_summary.csv")
    
    # Plot top 20
    plt.figure(figsize=(10, 8))
    sns.barplot(x=missing_df['Missing_Percent'].head(20), y=missing_df.head(20).index, palette='Reds_r')
    plt.title('Top 20 Features with Highest Missing Values (%)')
    plt.xlabel('Percentage Missing')
    plt.ylabel('Feature')
    plt.savefig(f"{FIG_DIR}/missing_values_top20.png", bbox_inches='tight')
    plt.close()

def analyze_correlations(df: pd.DataFrame):
    """Finds top positive and negative correlations with the target."""
    logging.info("Analyzing Correlations with Target...")
    
    # Only correlate numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['TARGET'].sort_values()
    
    # Save full correlations
    correlations.to_csv(f"{STATS_DIR}/target_correlations.csv")
    
    # Extract top 10 positive and top 10 negative (excluding TARGET itself)
    top_pos = correlations.tail(11).head(10)
    top_neg = correlations.head(10)
    top_corr = pd.concat([top_neg, top_pos])
    
    plt.figure(figsize=(8, 8))
    top_corr.plot(kind='barh', color=np.where(top_corr > 0, 'crimson', 'steelblue'))
    plt.title('Top 20 Correlations with Target')
    plt.xlabel('Pearson Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.savefig(f"{FIG_DIR}/top_correlations.png", bbox_inches='tight')
    plt.close()

def analyze_time_based_features(df: pd.DataFrame):
    """Analyzes pseudo-time features like age (DAYS_BIRTH)."""
    logging.info("Analyzing Time-based Features (DAYS_BIRTH)...")
    
    # Home Credit records time backwards from application date (negative values)
    # Convert DAYS_BIRTH to Age in Years
    df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365.25
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df.loc[df['TARGET'] == 0, 'AGE_YEARS'], label='Target == 0 (Repaid)', fill=True, alpha=0.5)
    sns.kdeplot(df.loc[df['TARGET'] == 1, 'AGE_YEARS'], label='Target == 1 (Default)', fill=True, alpha=0.5)
    
    plt.title('Distribution of Client Age by Target')
    plt.xlabel('Age (Years)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{FIG_DIR}/age_distribution_by_target.png", bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(RAW_DATA_PATH):
        logging.error(f"Dataset not found at {RAW_DATA_PATH}. Please download it first.")
        return
        
    df = load_data(RAW_DATA_PATH)
    
    analyze_target_distribution(df)
    analyze_missing_values(df)
    analyze_correlations(df)
    analyze_time_based_features(df)
    
    logging.info("EDA Complete. Check 'reports/figures' and 'reports/stats' for outputs.")

if __name__ == "__main__":
    main()