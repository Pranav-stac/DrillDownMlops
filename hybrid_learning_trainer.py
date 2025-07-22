import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta
import logging
from config import *

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, save_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Only XGBoost model will be used.")
    TENSORFLOW_AVAILABLE = False

class HybridLearningTrainer:
    def __init__(self):
        self.xgb_models = {}
        self.nn_models = {}
        self.scalers = {}
        self.feature_columns = []
        self.sequence_length = 24  # Use 24 hours of history for LSTM
        self.daily_performance = []
        self.learning_progress = []
        
        # Setup logging
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
        self.logger = logging.getLogger(__name__)
        
        # XGBoost parameters
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 50,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1,
            'reg_lambda': 1,
            'min_child_weight': 3,
            'random_state': 42
        }
        
        # Neural Network parameters
        self.nn_params = {
            'lstm_units': [64, 32],
            'dense_units': [16],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        }
        
        # Ensemble weights
        self.ensemble_weights = {
            'xgb': 0.6,
            'nn': 0.4
        }
        
    def load_and_prepare_data(self, data_path="enriched_sales_data_2023_2025.csv"):
        """Load and prepare data with proper aggregation"""
        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')
        
        # Filter out 2023 data - only use 2024 onwards
        original_year_count = len(df)
        df = df[df['date'].dt.year >= 2024]
        self.logger.info(f"Filtered to data from 2024 onwards: {len(df)} records (excluded {original_year_count - len(df)} records from 2023)")
        
        # Filter only dine-in orders
        if 'is_dine_in' in df.columns:
            original_count = len(df)
            df = df[df['is_dine_in'] == True]
            self.logger.info(f"Filtered to {len(df)} dine-in records (excluded {original_count - len(df)} online orders)")
        
        # Aggregate to hourly level properly
        hourly_df = df.groupby(['datetime']).agg({
            'final_total': 'sum',
            'quantity': 'sum',
            'item_name': 'count',  # customer count
            'temp': 'first',
            'humidity': 'first',
            'precip': 'first',
            'windspeed': 'first',
            'pressure': 'first',
            'cloudcover': 'first',
            'has_festival': 'first',
            'day_of_week': 'first',
            'month': 'first',
            'is_weekend': 'first'
        }).reset_index()
        
        # Rename columns
        hourly_df.columns = ['datetime', 'hourly_sales', 'hourly_quantity', 'customer_count',
                           'temp', 'humidity', 'precip', 'windspeed', 'pressure', 'cloudcover',
                           'has_festival', 'day_of_week', 'month', 'is_weekend']
        
        # Add time features
        hourly_df['hour'] = hourly_df['datetime'].dt.hour
        hourly_df['date'] = hourly_df['datetime'].dt.date
        
        # Fill missing values
        hourly_df = hourly_df.fillna(0)
        
        # Sort by datetime
        hourly_df = hourly_df.sort_values('datetime')
        
        self.logger.info(f"Created {len(hourly_df)} hourly records")
        return hourly_df
    
    def create_features(self, df):
        """Create features with proper lag calculations"""
        df = df.copy().sort_values('datetime')
        
        # Basic time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weather features
        df['temp_normalized'] = (df['temp'] - df['temp'].mean()) / (df['temp'].std() + 1e-8)
        df['humidity_normalized'] = (df['humidity'] - df['humidity'].mean()) / (df['humidity'].std() + 1e-8)
        
        # Lag features (properly calculated)
        df['sales_lag_1h'] = df['hourly_sales'].shift(1)
        df['sales_lag_24h'] = df['hourly_sales'].shift(24)
        df['sales_lag_168h'] = df['hourly_sales'].shift(168)  # 1 week
        
        df['customers_lag_1h'] = df['customer_count'].shift(1)
        df['customers_lag_24h'] = df['customer_count'].shift(24)
        
        # Rolling features
        df['sales_rolling_mean_7h'] = df['hourly_sales'].rolling(window=7, min_periods=1).mean()
        df['sales_rolling_mean_24h'] = df['hourly_sales'].rolling(window=24, min_periods=1).mean()
        
        # Same hour last week
        df['same_hour_last_week'] = df.groupby('hour')['hourly_sales'].shift(168)
        
        # Fill NaN values for lag features
        lag_columns = ['sales_lag_1h', 'sales_lag_24h', 'sales_lag_168h', 
                      'customers_lag_1h', 'customers_lag_24h', 
                      'sales_rolling_mean_7h', 'sales_rolling_mean_24h', 'same_hour_last_week']
        
        for col in lag_columns:
            df[col] = df[col].fillna(df['hourly_sales'].mean() if 'sales' in col else df['customer_count'].mean())
        
        # Feature columns
        self.feature_columns = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'has_festival',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'temp', 'humidity', 'precip', 'windspeed', 'pressure', 'cloudcover',
            'temp_normalized', 'humidity_normalized',
            'sales_lag_1h', 'sales_lag_24h', 'sales_lag_168h',
            'customers_lag_1h', 'customers_lag_24h',
            'sales_rolling_mean_7h', 'sales_rolling_mean_24h', 'same_hour_last_week'
        ]
        
        return df
    
    def prepare_sequence_data(self, df, target_col, lookback=24):
        """Prepare sequence data for LSTM"""
        if not TENSORFLOW_AVAILABLE:
            return None, None
            
        # Create sequences
        X, y = [], []
        
        for i in range(lookback, len(df)):
            # Get sequence of features
            sequence = df[self.feature_columns].iloc[i-lookback:i].values
            target = df[target_col].iloc[i]
            
            X.append(sequence)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def create_lstm_model(self, input_shape, output_size=1):
        """Create LSTM model for time series prediction"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        model = Sequential([
            LSTM(self.nn_params['lstm_units'][0], return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(self.nn_params['dropout_rate']),
            LSTM(self.nn_params['lstm_units'][1]),
            BatchNormalization(),
            Dropout(self.nn_params['dropout_rate']),
            Dense(self.nn_params['dense_units'][0], activation='relu'),
            Dense(output_size)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.nn_params['learning_rate']),
            loss='mean_squared_error'
        )
        
        return model
    
    def simulate_historical_learning(self, data_path="enriched_sales_data_2023_2025.csv"):
        """Simulate day-by-day learning with hybrid approach"""
        print("\n" + "="*100)
        print("🎯 HYBRID LEARNING: XGBOOST + NEURAL NETWORK (2024-2025 DATA ONLY)")
        print("="*100)
        
        # Load and prepare data
        df = self.load_and_prepare_data(data_path)
        df = self.create_features(df)
        
        # Get unique dates
        unique_dates = sorted(df['date'].unique())
        
        # Start with more training data
        min_training_days = 60
        start_idx = min_training_days
        
        print(f"📅 Date Range: {unique_dates[0]} to {unique_dates[-1]}")
        print(f"🎓 Initial Training Period: {min_training_days} days")
        print(f"🔄 Learning Days: {len(unique_dates) - min_training_days}")
        
        # Initial training
        initial_data = df[df['date'] <= unique_dates[start_idx-1]]
        self.initial_training(initial_data)
        
        # Day-by-day learning
        for day_idx in range(start_idx, len(unique_dates)):
            current_date = unique_dates[day_idx-1]
            prediction_date = unique_dates[day_idx]
            
            print(f"\n📅 Day {day_idx - start_idx + 1}: {current_date} → Predicting {prediction_date}")
            print("-" * 60)
            
            # Get training data up to current date
            training_data = df[df['date'] <= current_date]
            
            # Get actual data for prediction date
            actual_data = df[df['date'] == prediction_date]
            
            if len(actual_data) == 0:
                print("❌ No actual data available")
                continue
            
            # Make predictions
            predictions = self.predict_day(training_data, prediction_date)
            
            # Compare with actual values
            performance = self.evaluate_predictions(predictions, actual_data, prediction_date)
            
            # Learn from actual values
            self.learn_from_actual(actual_data)
            
            # Store performance
            self.daily_performance.append(performance)
            
            # Print results
            self.print_daily_performance(performance)
            
            # Update learning progress
            self.update_learning_progress(day_idx - start_idx + 1, performance)
        
        # Print final summary
        self.print_final_summary()
        
        # Save results
        self.save_learning_results()
        
        return True
    
    def initial_training(self, training_data):
        """Initial training with first batch of data"""
        print("🎓 Initial Model Training...")
        
        targets = ['hourly_sales', 'customer_count']
        
        for target_col in targets:
            if target_col in training_data.columns:
                # Train XGBoost model
                self.train_xgboost_model(training_data, target_col)
                
                # Train neural network model if TensorFlow is available
                if TENSORFLOW_AVAILABLE:
                    self.train_nn_model(training_data, target_col)
        
        print(f"🎯 Initial training completed with {len(self.xgb_models)} XGBoost models and {len(self.nn_models)} Neural Network models")
    
    def train_xgboost_model(self, training_data, target_col):
        """Train XGBoost model"""
        X = training_data[self.feature_columns].fillna(0)
        y = training_data[target_col].fillna(0)
        
        # Remove invalid samples but keep zeros
        valid_idx = (~y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) > 50:
            # Apply log transformation
            y_transformed = np.log1p(y)
            
            # Create and fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[target_col] = scaler
            
            # Train model
            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(X_scaled, y_transformed)
            self.xgb_models[target_col] = model
            print(f"✅ XGBoost {target_col} model trained with {len(X)} samples")
    
    def train_nn_model(self, training_data, target_col):
        """Train neural network model"""
        if not TENSORFLOW_AVAILABLE:
            return
            
        # Prepare sequence data
        X_seq, y_seq = self.prepare_sequence_data(training_data, target_col, self.sequence_length)
        
        if X_seq is not None and len(X_seq) > 50:
            # Apply log transformation
            y_transformed = np.log1p(y_seq)
            
            # Create model
            input_shape = (X_seq.shape[1], X_seq.shape[2])
            model = self.create_lstm_model(input_shape)
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            # Train model
            model.fit(
                X_seq, y_transformed,
                validation_split=0.2,
                epochs=self.nn_params['epochs'],
                batch_size=self.nn_params['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )
            
            self.nn_models[target_col] = model
            print(f"✅ Neural Network {target_col} model trained with {len(X_seq)} sequences")
    
    def predict_day(self, training_data, prediction_date):
        """Make predictions for a specific day"""
        prediction_hours = range(24)
        predictions = []
        
        for hour in prediction_hours:
            # Create feature vector for this hour
            features = self.create_prediction_features(prediction_date, hour, training_data)
            
            if features is not None:
                hour_predictions = {}
                
                for target_col in self.xgb_models.keys():
                    try:
                        # Get XGBoost prediction
                        xgb_pred = self.predict_xgboost(features, target_col)
                        
                        # Get neural network prediction if available
                        nn_pred = self.predict_nn(training_data, prediction_date, hour, target_col)
                        
                        # Ensemble predictions
                        if nn_pred is not None:
                            # Weighted average
                            final_pred = (xgb_pred * self.ensemble_weights['xgb'] + 
                                         nn_pred * self.ensemble_weights['nn'])
                        else:
                            final_pred = xgb_pred
                        
                        hour_predictions[target_col] = max(0, final_pred)
                    except Exception as e:
                        self.logger.error(f"Error predicting {target_col}: {str(e)}")
                        hour_predictions[target_col] = 0
                
                predictions.append({
                    'date': prediction_date,
                    'hour': hour,
                    'predictions': hour_predictions
                })
        
        return predictions
    
    def predict_xgboost(self, features, target_col):
        """Make prediction with XGBoost model"""
        if target_col not in self.xgb_models:
            return 0
            
        # Scale features
        X = features[self.feature_columns].fillna(0)
        X_scaled = self.scalers[target_col].transform(X)
        
        # Predict (log-transformed)
        pred_log = self.xgb_models[target_col].predict(X_scaled)[0]
        
        # Transform back from log
        pred = np.expm1(pred_log)
        
        return pred
    
    def predict_nn(self, training_data, prediction_date, hour, target_col):
        """Make prediction with neural network model"""
        if not TENSORFLOW_AVAILABLE or target_col not in self.nn_models:
            return None
            
        try:
            # Get sequence data leading up to this prediction
            recent_data = training_data.tail(self.sequence_length)
            
            if len(recent_data) < self.sequence_length:
                return None
                
            # Create sequence
            X_seq = recent_data[self.feature_columns].values.reshape(1, self.sequence_length, -1)
            
            # Predict (log-transformed)
            pred_log = self.nn_models[target_col].predict(X_seq, verbose=0)[0][0]
            
            # Transform back from log
            pred = np.expm1(pred_log)
            
            return pred
        except Exception as e:
            self.logger.error(f"Neural network prediction error: {str(e)}")
            return None
    
    def create_prediction_features(self, prediction_date, hour, training_data):
        """Create features for prediction with improved accuracy"""
        try:
            # Get data from the same hour on previous days
            same_hour_data = training_data[training_data['hour'] == hour].tail(30)
            
            # Get data from the previous day, same hour
            prev_day = prediction_date - timedelta(days=1)
            prev_day_data = training_data[
                (training_data['date'] == prev_day) & 
                (training_data['hour'] == hour)
            ]
            
            # Get data from the same day last week
            last_week_day = prediction_date - timedelta(days=7)
            last_week_data = training_data[
                (training_data['date'] == last_week_day) & 
                (training_data['hour'] == hour)
            ]
            
            # Create base feature vector
            base_features = {
                'hour': hour,
                'day_of_week': prediction_date.weekday(),
                'month': prediction_date.month,
                'is_weekend': 1 if prediction_date.weekday() >= 5 else 0,
            }
            
            # Add weather features - use average of recent days at same hour
            weather_cols = ['temp', 'humidity', 'precip', 'windspeed', 'pressure', 'cloudcover']
            for col in weather_cols:
                if col in same_hour_data.columns and len(same_hour_data) > 0:
                    base_features[col] = same_hour_data[col].mean()
                else:
                    base_features[col] = training_data[col].mean() if col in training_data.columns else 0
            
            # Add festival features
            base_features['has_festival'] = 0
            
            # Check if this date is a festival from training data
            festival_dates = training_data[training_data['has_festival'] > 0]['date'].unique()
            if prediction_date in festival_dates:
                base_features['has_festival'] = 1
            
            # Add lag features - much more accurate now
            # Previous hour
            prev_hour_data = training_data[training_data['datetime'] == 
                                         datetime.combine(prediction_date, 
                                                        datetime.min.time().replace(hour=(hour-1) % 24))]
            
            if len(prev_hour_data) > 0:
                base_features['sales_lag_1h'] = prev_hour_data['hourly_sales'].iloc[0]
                base_features['customers_lag_1h'] = prev_hour_data['customer_count'].iloc[0]
            elif len(same_hour_data) > 0:
                base_features['sales_lag_1h'] = same_hour_data['hourly_sales'].mean()
                base_features['customers_lag_1h'] = same_hour_data['customer_count'].mean()
            else:
                base_features['sales_lag_1h'] = training_data['hourly_sales'].mean()
                base_features['customers_lag_1h'] = training_data['customer_count'].mean()
            
            # Same hour yesterday
            if len(prev_day_data) > 0:
                base_features['sales_lag_24h'] = prev_day_data['hourly_sales'].iloc[0]
                base_features['customers_lag_24h'] = prev_day_data['customer_count'].iloc[0]
            else:
                base_features['sales_lag_24h'] = base_features['sales_lag_1h']
                base_features['customers_lag_24h'] = base_features['customers_lag_1h']
            
            # Same hour last week
            if len(last_week_data) > 0:
                base_features['sales_lag_168h'] = last_week_data['hourly_sales'].iloc[0]
                base_features['same_hour_last_week'] = last_week_data['hourly_sales'].iloc[0]
            else:
                base_features['sales_lag_168h'] = base_features['sales_lag_24h']
                base_features['same_hour_last_week'] = base_features['sales_lag_24h']
            
            # Rolling averages - use actual data from same hour on previous days
            if len(same_hour_data) > 0:
                base_features['sales_rolling_mean_7h'] = same_hour_data['hourly_sales'].tail(7).mean()
                base_features['sales_rolling_mean_24h'] = same_hour_data['hourly_sales'].mean()
            else:
                base_features['sales_rolling_mean_7h'] = base_features['sales_lag_1h']
                base_features['sales_rolling_mean_24h'] = base_features['sales_lag_1h']
            
            # Add cyclical features
            base_features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            base_features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            base_features['day_sin'] = np.sin(2 * np.pi * prediction_date.weekday() / 7)
            base_features['day_cos'] = np.cos(2 * np.pi * prediction_date.weekday() / 7)
            base_features['month_sin'] = np.sin(2 * np.pi * prediction_date.month / 12)
            base_features['month_cos'] = np.cos(2 * np.pi * prediction_date.month / 12)
            
            # Add normalized weather features
            base_features['temp_normalized'] = (base_features['temp'] - training_data['temp'].mean()) / (training_data['temp'].std() + 1e-8)
            base_features['humidity_normalized'] = (base_features['humidity'] - training_data['humidity'].mean()) / (training_data['humidity'].std() + 1e-8)
            
            # Create DataFrame
            features_df = pd.DataFrame([base_features])
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            return None
    
    def evaluate_predictions(self, predictions, actual_data, prediction_date):
        """Evaluate predictions against actual data"""
        # Aggregate actual data by hour
        actual_hourly = actual_data.groupby(actual_data['hour']).agg({
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
        # Update XGBoost models
        for target_col, model in self.xgb_models.items():
            if target_col in actual_data.columns:
                try:
                    X = actual_data[self.feature_columns].fillna(0)
                    y = actual_data[target_col].fillna(0)
                    
                    # Remove invalid samples
                    valid_idx = (~y.isna())
                    X = X[valid_idx]
                    y = y[valid_idx]
                    
                    if len(X) > 0:
                        # Apply log transformation
                        y_transformed = np.log1p(y)
                        
                        # Scale features
                        X_scaled = self.scalers[target_col].transform(X)
                        
                        # Incremental learning with reduced weight
                        sample_weight = np.ones(len(X)) * 0.5
                        model.fit(X_scaled, y_transformed, xgb_model=model.get_booster(), sample_weight=sample_weight)
                
                except Exception as e:
                    self.logger.error(f"Error in XGBoost incremental learning for {target_col}: {str(e)}")
        
        # Neural networks don't support true incremental learning in the same way
        # For production, you would need to implement a custom solution or retrain periodically
    
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
        print("🎉 HYBRID LEARNING COMPLETED!")
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
            
            performance_df.to_csv('hybrid_daily_learning_performance.csv', index=False)
            print(f"💾 Saved daily performance to hybrid_daily_learning_performance.csv")
        
        # Save learning progress
        if self.learning_progress:
            progress_df = pd.DataFrame(self.learning_progress)
            progress_df.to_csv('hybrid_learning_progress.csv', index=False)
            print(f"📈 Saved learning progress to hybrid_learning_progress.csv")
        
        # Save final models
        self.save_models()
    
    def save_models(self):
        """Save trained models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save XGBoost models
        for target_col, model in self.xgb_models.items():
            model_file = os.path.join(MODEL_PATH, f"{target_col}_xgb_model_{timestamp}.joblib")
            joblib.dump(model, model_file)
            
            # Also save as latest
            latest_file = os.path.join(MODEL_PATH, f"{target_col}_xgb_model_latest.joblib")
            joblib.dump(model, latest_file)
        
        # Save neural network models
        if TENSORFLOW_AVAILABLE:
            for target_col, model in self.nn_models.items():
                model_file = os.path.join(MODEL_PATH, f"{target_col}_nn_model_{timestamp}")
                model.save(model_file)
                
                # Also save as latest
                latest_file = os.path.join(MODEL_PATH, f"{target_col}_nn_model_latest")
                model.save(latest_file)
        
        # Save scalers
        for target_col, scaler in self.scalers.items():
            scaler_file = os.path.join(MODEL_PATH, f"{target_col}_scaler_latest.joblib")
            joblib.dump(scaler, scaler_file)
        
        # Save feature columns
        feature_file = os.path.join(MODEL_PATH, f"feature_columns_latest.joblib")
        joblib.dump(self.feature_columns, feature_file)
        
        print(f"🔧 Models saved with timestamp {timestamp}")

if __name__ == "__main__":
    trainer = HybridLearningTrainer()
    trainer.simulate_historical_learning()