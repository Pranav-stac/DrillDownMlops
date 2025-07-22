import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
from feature_engineering import FeatureEngineer

class SalesPredictionModel:
    def __init__(self):
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.feature_columns = []
        self.is_trained = False
        self.training_history = []
        self.daily_performance = []
        self.learning_curve = []
        
        # Setup logging
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
        self.logger = logging.getLogger(__name__)
        
    def load_and_prepare_data(self, data_path=None, end_date=None):
        """Load and prepare data for training"""
        if data_path is None:
            data_path = DATA_PATH
            
        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter data up to end_date if specified
        if end_date:
            df = df[df['date'] <= end_date]
        
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
    
    def split_data(self, df, test_days=30):
        """Split data into train and test sets based on time"""
        df = df.sort_values('datetime')
        
        # Calculate split date
        max_date = df['datetime'].max()
        split_date = max_date - timedelta(days=test_days)
        
        train_df = df[df['datetime'] <= split_date]
        test_df = df[df['datetime'] > split_date]
        
        self.logger.info(f"Train set: {len(train_df)} records (up to {split_date.date()})")
        self.logger.info(f"Test set: {len(test_df)} records (from {split_date.date()})")
        
        return train_df, test_df
    
    def train_model(self, df, target_col, model_name):
        """Train XGBoost model for a specific target"""
        # Prepare features and target
        X = df[self.feature_columns].fillna(0)
        y = df[target_col].fillna(0)
        
        # Remove rows where target is 0 or NaN for better training
        valid_idx = (y > 0) & (~y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 100:
            self.logger.warning(f"Not enough data for {target_col}: {len(X)} samples")
            return None
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train XGBoost model
        model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        self.logger.info(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
        
        return model
    
    def train_all_models(self, data_path=None, end_date=None):
        """Train all prediction models"""
        self.logger.info("Starting model training...")
        
        # Load and prepare data
        df = self.load_and_prepare_data(data_path, end_date)
        
        if len(df) < MIN_TRAINING_DAYS * 24:  # Minimum hours of data
            self.logger.error(f"Not enough data for training. Need at least {MIN_TRAINING_DAYS} days")
            return False
        
        # Train models for different targets
        targets = {
            'hourly_sales': 'Hourly Sales Model',
            'customer_count': 'Customer Count Model'
        }
        
        for target_col, model_name in targets.items():
            if target_col in df.columns:
                self.logger.info(f"Training {model_name}...")
                model = self.train_model(df, target_col, model_name)
                if model is not None:
                    self.models[target_col] = model
        
        if len(self.models) > 0:
            self.is_trained = True
            self.save_models()
            
            # Record training history
            self.training_history.append({
                'timestamp': datetime.now(),
                'data_end_date': end_date or df['datetime'].max(),
                'models_trained': list(self.models.keys()),
                'training_samples': len(df)
            })
            
            self.logger.info(f"Training completed. {len(self.models)} models trained.")
            return True
        else:
            self.logger.error("No models were successfully trained")
            return False
    
    def incremental_train(self, new_data_path=None, new_end_date=None):
        """Incrementally train models with new data"""
        self.logger.info("Starting incremental training...")
        
        if not self.is_trained:
            self.logger.info("No existing models found. Performing full training...")
            return self.train_all_models(new_data_path, new_end_date)
        
        # Load new data
        df = self.load_and_prepare_data(new_data_path, new_end_date)
        
        # Get only recent data for incremental training
        recent_date = df['datetime'].max() - timedelta(days=7)  # Last week of data
        recent_df = df[df['datetime'] >= recent_date]
        
        if len(recent_df) < 24:  # At least 1 day of data
            self.logger.warning("Not enough new data for incremental training")
            return False
        
        # Incrementally train each model
        for target_col, model in self.models.items():
            if target_col in recent_df.columns:
                self.logger.info(f"Incrementally training {target_col} model...")
                
                X_new = recent_df[self.feature_columns].fillna(0)
                y_new = recent_df[target_col].fillna(0)
                
                # Remove invalid samples
                valid_idx = (y_new > 0) & (~y_new.isna())
                X_new = X_new[valid_idx]
                y_new = y_new[valid_idx]
                
                if len(X_new) > 10:
                    # Continue training with new data
                    model.fit(X_new, y_new, xgb_model=model.get_booster())
        
        self.save_models()
        
        # Record incremental training
        self.training_history.append({
            'timestamp': datetime.now(),
            'type': 'incremental',
            'data_end_date': new_end_date or df['datetime'].max(),
            'training_samples': len(recent_df)
        })
        
        self.logger.info("Incremental training completed")
        return True
    
    def predict(self, input_features):
        """Make predictions using trained models"""
        if not self.is_trained:
            self.logger.error("Models not trained yet")
            return None
        
        # Ensure input has all required features, add missing ones with 0
        X = pd.DataFrame()
        for col in self.feature_columns:
            if col in input_features.columns:
                X[col] = input_features[col].fillna(0)
            else:
                X[col] = 0
        
        predictions = {}
        for target_col, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[target_col] = pred[0] if len(pred) == 1 else pred
            except Exception as e:
                self.logger.error(f"Error predicting {target_col}: {str(e)}")
                predictions[target_col] = 0
        
        return predictions
    
    def get_feature_importance(self, target_col='hourly_sales'):
        """Get feature importance for a specific model"""
        if target_col not in self.models:
            return None
        
        model = self.models[target_col]
        importance = model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_models(self):
        """Save trained models to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for target_col, model in self.models.items():
            model_file = os.path.join(MODEL_PATH, f"{target_col}_model_{timestamp}.joblib")
            joblib.dump(model, model_file)
            
            # Also save as latest
            latest_file = os.path.join(MODEL_PATH, f"{target_col}_model_latest.joblib")
            joblib.dump(model, latest_file)
        
        # Save feature columns
        feature_file = os.path.join(MODEL_PATH, f"feature_columns_{timestamp}.joblib")
        joblib.dump(self.feature_columns, feature_file)
        
        latest_feature_file = os.path.join(MODEL_PATH, "feature_columns_latest.joblib")
        joblib.dump(self.feature_columns, latest_feature_file)
        
        self.logger.info(f"Models saved with timestamp {timestamp}")
    
    def load_models(self):
        """Load latest trained models from disk"""
        try:
            # Load feature columns
            feature_file = os.path.join(MODEL_PATH, "feature_columns_latest.joblib")
            if os.path.exists(feature_file):
                self.feature_columns = joblib.load(feature_file)
            
            # Load models
            targets = ['hourly_sales', 'customer_count']
            for target_col in targets:
                model_file = os.path.join(MODEL_PATH, f"{target_col}_model_latest.joblib")
                if os.path.exists(model_file):
                    self.models[target_col] = joblib.load(model_file)
            
            if len(self.models) > 0:
                self.is_trained = True
                self.logger.info(f"Loaded {len(self.models)} models")
                return True
            else:
                self.logger.warning("No models found to load")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False
    
    def evaluate_model(self, test_data_path=None, test_end_date=None):
        """Evaluate model performance on test data"""
        if not self.is_trained:
            self.logger.error("Models not trained yet")
            return None
        
        # Load test data
        df = self.load_and_prepare_data(test_data_path, test_end_date)
        
        # Split data
        train_df, test_df = self.split_data(df)
        
        if len(test_df) == 0:
            self.logger.warning("No test data available")
            return None
        
        results = {}
        for target_col, model in self.models.items():
            if target_col in test_df.columns:
                X_test = test_df[self.feature_columns].fillna(0)
                y_test = test_df[target_col].fillna(0)
                
                # Remove invalid samples
                valid_idx = (y_test > 0) & (~y_test.isna())
                X_test = X_test[valid_idx]
                y_test = y_test[valid_idx]
                
                if len(X_test) > 0:
                    y_pred = model.predict(X_test)
                    
                    results[target_col] = {
                        'mae': mean_absolute_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'r2': r2_score(y_test, y_pred),
                        'samples': len(y_test)
                    }
        
        return results
    
    def train_with_progress_tracking(self, data_path=None, simulate_daily_learning=True):
        """Train model with detailed progress tracking and daily learning simulation"""
        self.logger.info("Starting training with progress tracking...")
        
        # Load and prepare data
        df = self.load_and_prepare_data(data_path)
        
        if len(df) < MIN_TRAINING_DAYS * 24:
            self.logger.error(f"Not enough data for training. Need at least {MIN_TRAINING_DAYS} days")
            return False
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        if simulate_daily_learning:
            return self.simulate_daily_learning(df)
        else:
            return self.train_all_models_with_tracking(df)
    
    def simulate_daily_learning(self, df):
        """Simulate daily learning process with historical data"""
        self.logger.info("Simulating daily learning process...")
        
        # Get date range
        start_date = df['datetime'].min().date()
        end_date = df['datetime'].max().date()
        
        # Start with minimum training days
        initial_end_date = start_date + timedelta(days=MIN_TRAINING_DAYS)
        
        current_date = initial_end_date
        day_count = 0
        
        print("\n" + "="*80)
        print("DAILY LEARNING SIMULATION")
        print("="*80)
        
        while current_date <= end_date:
            day_count += 1
            
            # Get training data up to current date
            train_data = df[df['datetime'].dt.date <= current_date]
            
            if len(train_data) < 24:  # Need at least 1 day of data
                current_date += timedelta(days=1)
                continue
            
            print(f"\nDay {day_count}: {current_date}")
            print("-" * 40)
            
            # Train models
            success = self.train_models_for_date(train_data, current_date)
            
            if success:
                # Get next day's data for evaluation
                next_date = current_date + timedelta(days=1)
                next_day_data = df[df['datetime'].dt.date == next_date]
                
                if len(next_day_data) > 0:
                    # Evaluate on next day
                    daily_performance = self.evaluate_daily_performance(next_day_data, current_date)
                    self.daily_performance.append(daily_performance)
                    
                    # Print daily results
                    self.print_daily_results(daily_performance)
                
                # Update learning curve
                self.update_learning_curve(current_date, len(train_data))
            
            current_date += timedelta(days=1)
        
        # Print overall summary
        self.print_training_summary()
        
        # Save performance data
        self.save_performance_data()
        
        # Create visualizations
        self.create_performance_visualizations()
        
        return True
    
    def train_models_for_date(self, train_data, current_date):
        """Train models for a specific date"""
        try:
            # Prepare features and targets
            targets = ['hourly_sales', 'customer_count']
            
            for target_col in targets:
                if target_col in train_data.columns:
                    # Prepare data
                    X = train_data[self.feature_columns].fillna(0)
                    y = train_data[target_col].fillna(0)
                    
                    # Remove invalid samples
                    valid_idx = (y > 0) & (~y.isna())
                    X = X[valid_idx]
                    y = y[valid_idx]
                    
                    if len(X) < 50:  # Minimum samples
                        continue
                    
                    # Train model
                    if target_col not in self.models:
                        # First time training
                        model = xgb.XGBRegressor(**XGBOOST_PARAMS)
                        model.fit(X, y)
                        self.models[target_col] = model
                    else:
                        # Incremental training
                        recent_data = train_data[train_data['datetime'].dt.date >= current_date - timedelta(days=7)]
                        if len(recent_data) > 10:
                            X_recent = recent_data[self.feature_columns].fillna(0)
                            y_recent = recent_data[target_col].fillna(0)
                            
                            valid_idx = (y_recent > 0) & (~y_recent.isna())
                            X_recent = X_recent[valid_idx]
                            y_recent = y_recent[valid_idx]
                            
                            if len(X_recent) > 5:
                                self.models[target_col].fit(X_recent, y_recent, 
                                                          xgb_model=self.models[target_col].get_booster())
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models for {current_date}: {str(e)}")
            return False
    
    def evaluate_daily_performance(self, next_day_data, training_date):
        """Evaluate model performance on next day's data"""
        performance = {
            'date': training_date,
            'evaluation_date': training_date + timedelta(days=1),
            'training_samples': 0,
            'evaluation_samples': len(next_day_data),
            'models': {}
        }
        
        for target_col, model in self.models.items():
            if target_col in next_day_data.columns:
                try:
                    X_eval = next_day_data[self.feature_columns].fillna(0)
                    y_true = next_day_data[target_col].fillna(0)
                    
                    # Remove invalid samples
                    valid_idx = (y_true > 0) & (~y_true.isna())
                    X_eval = X_eval[valid_idx]
                    y_true = y_true[valid_idx]
                    
                    if len(X_eval) > 0:
                        y_pred = model.predict(X_eval)
                        
                        mae = mean_absolute_error(y_true, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                        r2 = r2_score(y_true, y_pred)
                        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                        
                        performance['models'][target_col] = {
                            'mae': mae,
                            'rmse': rmse,
                            'r2': r2,
                            'mape': mape,
                            'samples': len(y_true),
                            'actual_total': y_true.sum(),
                            'predicted_total': y_pred.sum(),
                            'accuracy': max(0, 100 - mape)
                        }
                
                except Exception as e:
                    self.logger.error(f"Error evaluating {target_col}: {str(e)}")
        
        return performance
    
    def print_daily_results(self, performance):
        """Print daily performance results"""
        eval_date = performance['evaluation_date']
        
        for target_col, metrics in performance['models'].items():
            print(f"  {target_col.replace('_', ' ').title()}:")
            print(f"    Accuracy: {metrics['accuracy']:.1f}%")
            print(f"    MAE: {metrics['mae']:.2f}")
            print(f"    R²: {metrics['r2']:.3f}")
            print(f"    Actual Total: {metrics['actual_total']:.0f}")
            print(f"    Predicted Total: {metrics['predicted_total']:.0f}")
            
            # Calculate prediction vs actual percentage
            if metrics['actual_total'] > 0:
                pred_accuracy = (1 - abs(metrics['predicted_total'] - metrics['actual_total']) / metrics['actual_total']) * 100
                print(f"    Total Prediction Accuracy: {pred_accuracy:.1f}%")
    
    def update_learning_curve(self, date, training_samples):
        """Update learning curve data"""
        if len(self.daily_performance) > 0:
            latest_performance = self.daily_performance[-1]
            
            avg_accuracy = 0
            model_count = 0
            
            for target_col, metrics in latest_performance['models'].items():
                avg_accuracy += metrics['accuracy']
                model_count += 1
            
            if model_count > 0:
                avg_accuracy = avg_accuracy / model_count
                
                self.learning_curve.append({
                    'date': date,
                    'training_samples': training_samples,
                    'avg_accuracy': avg_accuracy,
                    'evaluation_date': latest_performance['evaluation_date']
                })   
    def train_all_models_with_tracking(self, df):
        """Train all models with detailed tracking"""
        self.logger.info("Training all models with tracking...")
        
        # Split data for evaluation
        train_df, test_df = self.split_data(df)
        
        # Train models for different targets
        targets = {
            'hourly_sales': 'Hourly Sales Model',
            'customer_count': 'Customer Count Model'
        }
        
        for target_col, model_name in targets.items():
            if target_col in train_df.columns:
                self.logger.info(f"Training {model_name}...")
                model = self.train_model(train_df, target_col, model_name)
                if model is not None:
                    self.models[target_col] = model
        
        if len(self.models) > 0:
            self.is_trained = True
            
            # Evaluate on test data
            if len(test_df) > 0:
                results = {}
                for target_col, model in self.models.items():
                    if target_col in test_df.columns:
                        X_test = test_df[self.feature_columns].fillna(0)
                        y_test = test_df[target_col].fillna(0)
                        
                        # Remove invalid samples
                        valid_idx = (y_test > 0) & (~y_test.isna())
                        X_test = X_test[valid_idx]
                        y_test = y_test[valid_idx]
                        
                        if len(X_test) > 0:
                            y_pred = model.predict(X_test)
                            
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            r2 = r2_score(y_test, y_pred)
                            
                            print(f"{target_col.upper()} MODEL:")
                            print(f"MAE: {mae:.2f}")
                            print(f"RMSE: {rmse:.2f}")
                            print(f"R²: {r2:.3f}")
                            print(f"Samples: {len(y_test)}")
            
            self.save_models()
            return True
        else:
            self.logger.error("No models were successfully trained")
            return False
    
    def print_training_summary(self):
        """Print summary of training results"""
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        if len(self.daily_performance) > 0:
            # Calculate average metrics
            accuracies = []
            for perf in self.daily_performance:
                for target_col, metrics in perf['models'].items():
                    accuracies.append(metrics['accuracy'])
            
            avg_accuracy = np.mean(accuracies) if accuracies else 0
            
            print(f"Average Accuracy: {avg_accuracy:.2f}%")
            print(f"Days Simulated: {len(self.daily_performance)}")
            
            # Show improvement over time
            if len(self.daily_performance) >= 10:
                first_10 = []
                last_10 = []
                
                for perf in self.daily_performance[:10]:
                    for target_col, metrics in perf['models'].items():
                        first_10.append(metrics['accuracy'])
                
                for perf in self.daily_performance[-10:]:
                    for target_col, metrics in perf['models'].items():
                        last_10.append(metrics['accuracy'])
                
                first_10_avg = np.mean(first_10) if first_10 else 0
                last_10_avg = np.mean(last_10) if last_10 else 0
                
                improvement = last_10_avg - first_10_avg
                print(f"Improvement: {improvement:+.2f}% (first 10 days vs last 10 days)")
        
        print("="*80)
    
    def save_performance_data(self):
        """Save performance data to CSV files"""
        if len(self.daily_performance) > 0:
            # Flatten daily performance data
            performance_data = []
            for perf in self.daily_performance:
                for target_col, metrics in perf['models'].items():
                    performance_data.append({
                        'date': perf['date'],
                        'evaluation_date': perf['evaluation_date'],
                        'target': target_col,
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'r2': metrics['r2'],
                        'accuracy': metrics['accuracy'],
                        'actual_total': metrics['actual_total'],
                        'predicted_total': metrics['predicted_total']
                    })
            
            # Save to CSV
            performance_df = pd.DataFrame(performance_data)
            performance_df.to_csv('daily_performance.csv', index=False)
            self.logger.info("Performance data saved to daily_performance.csv")
            
            # Save learning curve
            if len(self.learning_curve) > 0:
                learning_df = pd.DataFrame(self.learning_curve)
                learning_df.to_csv('learning_curve.csv', index=False)
                self.logger.info("Learning curve data saved to learning_curve.csv")
    
    def create_performance_visualizations(self):
        """Create visualizations of model performance"""
        if len(self.daily_performance) == 0:
            return
            
        try:
            # Flatten daily performance data
            dates = []
            accuracies = []
            targets = []
            
            for perf in self.daily_performance:
                for target_col, metrics in perf['models'].items():
                    dates.append(perf['evaluation_date'])
                    accuracies.append(metrics['accuracy'])
                    targets.append(target_col)
            
            # Create DataFrame
            viz_df = pd.DataFrame({
                'date': dates,
                'accuracy': accuracies,
                'target': targets
            })
            
            # Plot accuracy over time
            plt.figure(figsize=(12, 6))
            
            for target in viz_df['target'].unique():
                target_df = viz_df[viz_df['target'] == target]
                plt.plot(target_df['date'], target_df['accuracy'], label=target)
            
            plt.title('Prediction Accuracy Over Time')
            plt.xlabel('Date')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save plot
            plt.savefig('accuracy_over_time.png')
            self.logger.info("Performance visualization saved to accuracy_over_time.png")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")