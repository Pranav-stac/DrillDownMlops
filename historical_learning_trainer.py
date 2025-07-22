import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from config import *
from feature_engineering import FeatureEngineer

class HistoricalLearningTrainer:
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.feature_columns = []
        self.daily_performance = []
        self.learning_progress = []
        
        # Setup logging
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
        self.logger = logging.getLogger(__name__)
        
    def load_and_prepare_data(self, data_path=None):
        """Load and prepare all data"""
        if data_path is None:
            data_path = DATA_PATH
            
        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        self.logger.info(f"Loaded {len(df)} total records")
        
        # Filter only dine-in orders (exclude online orders)
        if 'is_dine_in' in df.columns:
            original_count = len(df)
            df = df[df['is_dine_in'] == True]
            self.logger.info(f"Filtered to {len(df)} dine-in records (excluded {original_count - len(df)} online orders)")
        
        # Prepare features
        prepared_df = self.feature_engineer.prepare_features(df)
        self.feature_columns = self.feature_engineer.feature_columns
        
        return prepared_df
    
    def simulate_historical_learning(self, data_path=None):
        """Simulate day-by-day learning process with historical data"""
        print("\n" + "="*100)
        print("🎯 HISTORICAL CONTINUOUS LEARNING SIMULATION")
        print("="*100)
        print("📊 Training Strategy: Train today → Predict tomorrow → Learn from actual → Repeat")
        print("="*100)
        
        # Load all data
        df = self.load_and_prepare_data(data_path)
        df = df.sort_values('datetime')
        
        # Get unique dates
        unique_dates = sorted(df['datetime'].dt.date.unique())
        
        # Start with minimum training period
        min_training_days = 30
        start_training_idx = min_training_days
        
        print(f"📅 Date Range: {unique_dates[0]} to {unique_dates[-1]}")
        print(f"🔢 Total Days: {len(unique_dates)}")
        print(f"🎓 Initial Training Period: {min_training_days} days")
        print(f"🔄 Learning Days: {len(unique_dates) - min_training_days}")
        print("\n" + "-"*100)
        
        # Initialize models with first training period
        initial_training_data = df[df['datetime'].dt.date <= unique_dates[start_training_idx-1]]
        self.initial_training(initial_training_data)
        
        # Day-by-day learning simulation
        for day_idx in range(start_training_idx, len(unique_dates)):
            current_date = unique_dates[day_idx-1]  # Training date
            prediction_date = unique_dates[day_idx]  # Prediction date
            
            print(f"\n📅 Day {day_idx - start_training_idx + 1}: {current_date} → Predicting {prediction_date}")
            print("-" * 60)
            
            # Get training data up to current date
            training_data = df[df['datetime'].dt.date <= current_date]
            
            # Get actual data for prediction date
            actual_data = df[df['datetime'].dt.date == prediction_date]
            
            if len(actual_data) == 0:
                print("❌ No actual data available for this date")
                continue
            
            # Make predictions for prediction_date
            predictions = self.predict_day(training_data, prediction_date)
            
            # Compare with actual values
            performance = self.evaluate_predictions(predictions, actual_data, prediction_date)
            
            # Learn from actual values (incremental training)
            self.learn_from_actual(actual_data)
            
            # Store performance
            self.daily_performance.append(performance)
            
            # Print results
            self.print_daily_performance(performance)
            
            # Update learning progress
            self.update_learning_progress(day_idx - start_training_idx + 1, performance)
        
        # Print final summary
        self.print_final_summary()
        
        # Save results
        self.save_learning_results()
        
        return True
    
    def initial_training(self, training_data):
        """Initial training with first batch of data"""
        print("🎓 Initial Model Training...")
        
        # Improved XGBoost parameters to prevent overfitting
        improved_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 30,      # Reduced
            'max_depth': 3,          # Reduced
            'learning_rate': 0.05,   # Reduced
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 2,          # Increased regularization
            'reg_lambda': 2,         # Increased regularization
            'random_state': 42,
            'min_child_weight': 3    # Added to prevent overfitting
        }
        
        targets = ['hourly_sales', 'customer_count']
        
        for target_col in targets:
            if target_col in training_data.columns:
                X = training_data[self.feature_columns].fillna(0)
                y = training_data[target_col].fillna(0)
                
                # Remove invalid samples but keep zeros (they're valid)
                valid_idx = (~y.isna())
                X = X[valid_idx]
                y = y[valid_idx]
                
                if len(X) > 50:  # Reduced minimum requirement
                    # Apply log transformation to reduce extreme values
                    y_transformed = np.log1p(y)  # log(1 + y) to handle zeros
                    
                    model = xgb.XGBRegressor(**improved_params)
                    model.fit(X, y_transformed)
                    self.models[target_col] = model
                    print(f"✅ {target_col} model trained with {len(X)} samples (log-transformed)")
        
        print(f"🎯 Initial training completed with {len(self.models)} models")
    
    def predict_day(self, training_data, prediction_date):
        """Make predictions for a specific day"""
        # Get unique hours for the prediction date
        prediction_hours = range(24)  # Predict for all 24 hours
        
        predictions = []
        
        for hour in prediction_hours:
            # Create feature vector for this hour
            features = self.create_prediction_features(prediction_date, hour, training_data)
            
            if features is not None:
                hour_predictions = {}
                
                for target_col, model in self.models.items():
                    try:
                        X = features[self.feature_columns].fillna(0)
                        pred = model.predict(X)[0]
                        hour_predictions[target_col] = max(0, pred)  # Ensure non-negative
                    except Exception as e:
                        hour_predictions[target_col] = 0
                
                predictions.append({
                    'date': prediction_date,
                    'hour': hour,
                    'predictions': hour_predictions
                })
        
        return predictions
    
    def create_prediction_features(self, prediction_date, hour, training_data):
        """Create features for prediction with proper lag calculation"""
        try:
            # Get the exact same hour from previous days for better lag features
            same_hour_data = training_data[training_data['datetime'].dt.hour == hour].tail(10)
            
            if len(same_hour_data) == 0:
                return None
            
            # Get recent data for weather and other features
            recent_data = training_data.tail(24)  # Last 24 hours
            
            # Create base feature vector
            base_features = {
                'hour': hour,
                'day_of_week': prediction_date.weekday(),
                'month': prediction_date.month,
                'is_weekend': 1 if prediction_date.weekday() >= 5 else 0,
            }
            
            # Add weather features (use recent averages)
            weather_cols = ['temp', 'humidity', 'precip', 'windspeed', 'pressure', 'cloudcover']
            for col in weather_cols:
                if col in recent_data.columns and len(recent_data) > 0:
                    base_features[col] = recent_data[col].mean()
                else:
                    # Default weather values
                    defaults = {'temp': 28, 'humidity': 70, 'precip': 0, 'windspeed': 10, 'pressure': 1013, 'cloudcover': 50}
                    base_features[col] = defaults.get(col, 0)
            
            # Add festival features
            base_features['has_festival'] = 0
            
            # Calculate lag features more carefully
            if len(training_data) >= 1:
                # Use actual recent values
                base_features['sales_lag_1h'] = training_data['hourly_sales'].iloc[-1] if len(training_data) >= 1 else 0
                base_features['sales_lag_24h'] = training_data['hourly_sales'].iloc[-24] if len(training_data) >= 24 else base_features['sales_lag_1h']
                base_features['sales_lag_168h'] = training_data['hourly_sales'].iloc[-168] if len(training_data) >= 168 else base_features['sales_lag_24h']
                
                base_features['customers_lag_1h'] = training_data['customer_count'].iloc[-1] if len(training_data) >= 1 else 0
                base_features['customers_lag_24h'] = training_data['customer_count'].iloc[-24] if len(training_data) >= 24 else base_features['customers_lag_1h']
                
                # Rolling averages
                base_features['sales_rolling_mean_7h'] = training_data['hourly_sales'].tail(7).mean()
                base_features['sales_rolling_mean_24h'] = training_data['hourly_sales'].tail(24).mean()
                
                # Same hour last week (if available)
                if len(same_hour_data) >= 2:
                    base_features['same_hour_last_week'] = same_hour_data['hourly_sales'].iloc[-2]
                else:
                    base_features['same_hour_last_week'] = base_features['sales_rolling_mean_24h']
            else:
                # Fallback values
                for feature in ['sales_lag_1h', 'sales_lag_24h', 'sales_lag_168h', 'sales_rolling_mean_7h', 'sales_rolling_mean_24h', 'same_hour_last_week']:
                    base_features[feature] = 0
                for feature in ['customers_lag_1h', 'customers_lag_24h']:
                    base_features[feature] = 0
            
            # Add cyclical features
            base_features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            base_features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            base_features['day_sin'] = np.sin(2 * np.pi * prediction_date.weekday() / 7)
            base_features['day_cos'] = np.cos(2 * np.pi * prediction_date.weekday() / 7)
            base_features['month_sin'] = np.sin(2 * np.pi * prediction_date.month / 12)
            base_features['month_cos'] = np.cos(2 * np.pi * prediction_date.month / 12)
            
            # Create DataFrame
            features_df = pd.DataFrame([base_features])
            
            # Only include features that exist in the model
            available_features = [col for col in self.feature_columns if col in features_df.columns]
            
            # Add missing features with default values
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            return features_df[self.feature_columns]
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            return None
    
    def evaluate_predictions(self, predictions, actual_data, prediction_date):
        """Evaluate predictions against actual data"""
        # Aggregate actual data by hour
        actual_hourly = actual_data.groupby(actual_data['datetime'].dt.hour).agg({
            'hourly_sales': 'first',
            'customer_count': 'first'
        }).reset_index()
        actual_hourly.columns = ['hour', 'actual_sales', 'actual_customers']
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame([
            {
                'hour': p['hour'],
                'pred_sales': p['predictions'].get('hourly_sales', 0),
                'pred_customers': p['predictions'].get('customer_count', 0)
            }
            for p in predictions
        ])
        
        # Merge predictions with actual
        comparison = pd.merge(pred_df, actual_hourly, on='hour', how='outer').fillna(0)
        
        # Calculate metrics
        performance = {
            'date': prediction_date,
            'total_actual_sales': comparison['actual_sales'].sum(),
            'total_predicted_sales': comparison['pred_sales'].sum(),
            'total_actual_customers': comparison['actual_customers'].sum(),
            'total_predicted_customers': comparison['pred_customers'].sum(),
            'hourly_details': comparison.to_dict('records')
        }
        
        # Calculate accuracy metrics
        if performance['total_actual_sales'] > 0:
            sales_accuracy = max(0, 100 - abs(performance['total_predicted_sales'] - performance['total_actual_sales']) / performance['total_actual_sales'] * 100)
            performance['sales_accuracy'] = sales_accuracy
        else:
            performance['sales_accuracy'] = 0
        
        if performance['total_actual_customers'] > 0:
            customer_accuracy = max(0, 100 - abs(performance['total_predicted_customers'] - performance['total_actual_customers']) / performance['total_actual_customers'] * 100)
            performance['customer_accuracy'] = customer_accuracy
        else:
            performance['customer_accuracy'] = 0
        
        return performance
    
    def learn_from_actual(self, actual_data):
        """Learn from actual data (incremental training)"""
        for target_col, model in self.models.items():
            if target_col in actual_data.columns:
                try:
                    X = actual_data[self.feature_columns].fillna(0)
                    y = actual_data[target_col].fillna(0)
                    
                    # Remove invalid samples
                    valid_idx = (y > 0) & (~y.isna())
                    X = X[valid_idx]
                    y = y[valid_idx]
                    
                    if len(X) > 0:
                        # Incremental learning
                        model.fit(X, y, xgb_model=model.get_booster())
                
                except Exception as e:
                    self.logger.error(f"Error in incremental learning for {target_col}: {str(e)}")
    
    def print_daily_performance(self, performance):
        """Print daily performance results"""
        date = performance['date']
        sales_acc = performance['sales_accuracy']
        customer_acc = performance['customer_accuracy']
        
        print(f"📊 Sales: ₹{performance['total_actual_sales']:.0f} (actual) vs ₹{performance['total_predicted_sales']:.0f} (predicted)")
        print(f"👥 Customers: {performance['total_actual_customers']:.0f} (actual) vs {performance['total_predicted_customers']:.0f} (predicted)")
        print(f"🎯 Accuracy: Sales {sales_acc:.1f}% | Customers {customer_acc:.1f}%")
        
        # Show accuracy color coding
        if sales_acc >= 80:
            print("🟢 Excellent prediction!")
        elif sales_acc >= 60:
            print("🟡 Good prediction")
        else:
            print("🔴 Needs improvement")
    
    def update_learning_progress(self, day_num, performance):
        """Update learning progress tracking"""
        self.learning_progress.append({
            'day': day_num,
            'sales_accuracy': performance['sales_accuracy'],
            'customer_accuracy': performance['customer_accuracy'],
            'total_sales': performance['total_actual_sales']
        })
    
    def print_final_summary(self):
        """Print final learning summary"""
        print("\n" + "="*100)
        print("🎉 HISTORICAL LEARNING COMPLETED!")
        print("="*100)
        
        if len(self.daily_performance) > 0:
            avg_sales_acc = np.mean([p['sales_accuracy'] for p in self.daily_performance])
            avg_customer_acc = np.mean([p['customer_accuracy'] for p in self.daily_performance])
            
            print(f"📈 Average Sales Accuracy: {avg_sales_acc:.1f}%")
            print(f"👥 Average Customer Accuracy: {avg_customer_acc:.1f}%")
            
            # Show improvement over time
            if len(self.learning_progress) >= 10:
                first_10_avg = np.mean([p['sales_accuracy'] for p in self.learning_progress[:10]])
                last_10_avg = np.mean([p['sales_accuracy'] for p in self.learning_progress[-10:]])
                improvement = last_10_avg - first_10_avg
                
                print(f"🚀 Learning Improvement: {improvement:+.1f}% (first 10 days vs last 10 days)")
                
                if improvement > 5:
                    print("🎯 Model is learning and improving!")
                elif improvement > 0:
                    print("📊 Model shows steady learning")
                else:
                    print("⚠️ Model may need tuning")
        
        print("="*100)
    
    def save_learning_results(self):
        """Save learning results to files"""
        # Save daily performance
        if self.daily_performance:
            performance_df = pd.DataFrame([
                {
                    'date': p['date'],
                    'sales_accuracy': p['sales_accuracy'],
                    'customer_accuracy': p['customer_accuracy'],
                    'actual_sales': p['total_actual_sales'],
                    'predicted_sales': p['total_predicted_sales'],
                    'actual_customers': p['total_actual_customers'],
                    'predicted_customers': p['total_predicted_customers']
                }
                for p in self.daily_performance
            ])
            
            performance_df.to_csv('daily_learning_performance.csv', index=False)
            print(f"💾 Saved daily performance to daily_learning_performance.csv")
        
        # Save learning progress
        if self.learning_progress:
            progress_df = pd.DataFrame(self.learning_progress)
            progress_df.to_csv('learning_progress.csv', index=False)
            print(f"📈 Saved learning progress to learning_progress.csv")
        
        # Save final models
        self.save_models()
    
    def save_models(self):
        """Save trained models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for target_col, model in self.models.items():
            model_file = os.path.join(MODEL_PATH, f"{target_col}_historical_model_{timestamp}.joblib")
            joblib.dump(model, model_file)
            
            # Also save as latest
            latest_file = os.path.join(MODEL_PATH, f"{target_col}_model_latest.joblib")
            joblib.dump(model, latest_file)
        
        # Save feature columns
        feature_file = os.path.join(MODEL_PATH, f"feature_columns_latest.joblib")
        joblib.dump(self.feature_columns, feature_file)
        
        print(f"🔧 Models saved with timestamp {timestamp}")

if __name__ == "__main__":
    trainer = HistoricalLearningTrainer()
    trainer.simulate_historical_learning()