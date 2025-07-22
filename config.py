# Configuration settings for the sales prediction system

import os
from datetime import datetime, timedelta

# File paths
DATA_PATH = "enriched_sales_data_2023_2025.csv"
MODEL_PATH = "models/"
LOGS_PATH = "logs/"
PREDICTIONS_PATH = "predictions/"

# Create directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(PREDICTIONS_PATH, exist_ok=True)

# Model parameters
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Feature engineering parameters
LOOKBACK_DAYS = 30  # Days to look back for rolling features
LAG_HOURS = [1, 24, 168]  # 1 hour, 1 day, 1 week lags

# Training parameters
TRAIN_TEST_SPLIT_DAYS = 30  # Last 30 days for testing
MIN_TRAINING_DAYS = 90  # Minimum days needed for training
RETRAIN_FREQUENCY = 1  # Retrain every N days

# Dashboard settings
DEFAULT_PREDICTION_HOURS = 24  # Default hours to predict ahead
CONFIDENCE_INTERVAL = 0.95

# BigQuery settings (if using)
BIGQUERY_PROJECT = "drilldown-big-query-database"
BIGQUERY_DATASET = "Ettarra_Juhu"
BIGQUERY_TABLE = "SALES_MASTERDATA"
CREDENTIALS_PATH = "creds.json"

# Weather API settings
WEATHER_LAT = 19.0948  # Mumbai Santacruz latitude
WEATHER_LON = 72.8471  # Mumbai Santacruz longitude

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"