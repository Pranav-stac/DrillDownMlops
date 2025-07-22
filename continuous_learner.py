import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import schedule
import os
from config import *
from model_trainer import SalesPredictionModel
from predictor import SalesPredictor
from enrich_sales_data import get_sales_data_from_bigquery

class ContinuousLearner:
    def __init__(self):
        self.model_trainer = SalesPredictionModel()
        self.predictor = SalesPredictor()
        
        # Setup logging
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_history = []
        self.last_training_date = None
        
    def fetch_latest_data(self, days_back=7):
        """Fetch latest sales data from BigQuery"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            self.logger.info(f"Fetching data from {start_date} to {end_date}")
            
            # Fetch from BigQuery
            sales_df = get_sales_data_from_bigquery(
                CREDENTIALS_PATH, 
                start_date.strftime('%Y-%m-%d'), 
                end_date.strftime('%Y-%m-%d')
            )
            
            if sales_df is not None and len(sales_df) > 0:
                # Save to temporary file
                temp_file = f"temp_sales_{datetime.now().strftime('%Y%m%d')}.csv"
                sales_df.to_csv(temp_file, index=False)
                
                self.logger.info(f"Fetched {len(sales_df)} new records")
                return temp_file
            else:
                self.logger.warning("No new data fetched")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching latest data: {str(e)}")
            return None
    
    def evaluate_predictions(self, actual_data_file, prediction_date):
        """Evaluate previous predictions against actual data"""
        try:
            # Load actual data
            actual_df = pd.read_csv(actual_data_file)
            actual_df['date'] = pd.to_datetime(actual_df['date']).dt.date
            
            # Filter for prediction date
            actual_day_data = actual_df[actual_df['date'] == prediction_date]
            
            if len(actual_day_data) == 0:
                self.logger.warning(f"No actual data found for {prediction_date}")
                return None
            
            # Calculate actual hourly sales
            actual_hourly = actual_day_data.groupby('hour').agg({
                'final_total': 'sum',
                'item_name': 'count'
            }).reset_index()
            actual_hourly.columns = ['hour', 'actual_sales', 'actual_customers']
            
            # Load previous predictions if they exist
            prediction_file = os.path.join(PREDICTIONS_PATH, f"predictions_{prediction_date.strftime('%Y%m%d')}.csv")
            
            if not os.path.exists(prediction_file):
                self.logger.warning(f"No predictions found for {prediction_date}")
                return None
            
            predicted_df = pd.read_csv(prediction_file)
            
            # Merge actual and predicted
            comparison = pd.merge(actual_hourly, predicted_df, on='hour', how='inner')
            
            if len(comparison) == 0:
                return None
            
            # Calculate metrics
            sales_mae = np.mean(np.abs(comparison['actual_sales'] - comparison['predicted_sales']))
            sales_mape = np.mean(np.abs((comparison['actual_sales'] - comparison['predicted_sales']) / comparison['actual_sales'])) * 100
            
            customers_mae = np.mean(np.abs(comparison['actual_customers'] - comparison['predicted_customers']))
            
            evaluation_result = {
                'date': prediction_date,
                'sales_mae': sales_mae,
                'sales_mape': sales_mape,
                'customers_mae': customers_mae,
                'total_actual_sales': comparison['actual_sales'].sum(),
                'total_predicted_sales': comparison['predicted_sales'].sum(),
                'accuracy_score': max(0, 100 - sales_mape)
            }
            
            self.performance_history.append(evaluation_result)
            
            self.logger.info(f"Evaluation for {prediction_date}: Sales MAPE: {sales_mape:.2f}%, Accuracy: {evaluation_result['accuracy_score']:.2f}%")
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {str(e)}")
            return None
    
    def should_retrain(self, evaluation_result=None):
        """Determine if model should be retrained"""
        # Retrain if it's been more than RETRAIN_FREQUENCY days
        if self.last_training_date is None:
            return True
        
        days_since_training = (datetime.now().date() - self.last_training_date).days
        if days_since_training >= RETRAIN_FREQUENCY:
            return True
        
        # Retrain if performance has degraded
        if evaluation_result and evaluation_result['accuracy_score'] < 70:
            self.logger.info("Performance degraded, triggering retrain")
            return True
        
        # Retrain if we have enough new data
        if len(self.performance_history) >= 7:  # Weekly retraining
            return True
        
        return False
    
    def daily_learning_cycle(self):
        """Execute daily learning cycle"""
        self.logger.info("Starting daily learning cycle...")
        
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        try:
            # Step 1: Fetch latest data
            latest_data_file = self.fetch_latest_data(days_back=7)
            
            if latest_data_file is None:
                self.logger.warning("No new data available, skipping learning cycle")
                return
            
            # Step 2: Evaluate yesterday's predictions
            evaluation_result = self.evaluate_predictions(latest_data_file, yesterday)
            
            # Step 3: Decide if retraining is needed
            if self.should_retrain(evaluation_result):
                self.logger.info("Retraining model with new data...")
                
                # Combine with existing data
                if os.path.exists(DATA_PATH):
                    existing_df = pd.read_csv(DATA_PATH)
                    new_df = pd.read_csv(latest_data_file)
                    
                    # Combine and remove duplicates
                    combined_df = pd.concat([existing_df, new_df]).drop_duplicates()
                    combined_df.to_csv(DATA_PATH, index=False)
                    
                    self.logger.info(f"Combined dataset now has {len(combined_df)} records")
                
                # Retrain model
                success = self.model_trainer.incremental_train(DATA_PATH, today)
                
                if success:
                    self.last_training_date = today
                    self.logger.info("Model retrained successfully")
                    
                    # Reload predictor with new model
                    self.predictor = SalesPredictor()
                else:
                    self.logger.error("Model retraining failed")
            
            # Step 4: Generate predictions for today
            self.generate_daily_predictions(today)
            
            # Step 5: Clean up temporary files
            if os.path.exists(latest_data_file):
                os.remove(latest_data_file)
            
            # Step 6: Save performance history
            self.save_performance_history()
            
            self.logger.info("Daily learning cycle completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in daily learning cycle: {str(e)}")
    
    def generate_daily_predictions(self, target_date):
        """Generate and save predictions for a specific date"""
        try:
            self.logger.info(f"Generating predictions for {target_date}")
            
            # Generate daily predictions
            daily_prediction = self.predictor.predict_daily_sales(target_date)
            
            if daily_prediction is None:
                self.logger.error("Failed to generate daily predictions")
                return
            
            # Save hourly predictions to CSV
            hourly_data = []
            for hour_pred in daily_prediction['hourly_predictions']:
                hourly_data.append({
                    'hour': hour_pred['hour'],
                    'predicted_sales': hour_pred['predictions']['hourly_sales'],
                    'predicted_customers': hour_pred['predictions']['customer_count'],
                    'confidence': hour_pred['confidence']
                })
            
            hourly_df = pd.DataFrame(hourly_data)
            prediction_file = os.path.join(PREDICTIONS_PATH, f"predictions_{target_date.strftime('%Y%m%d')}.csv")
            hourly_df.to_csv(prediction_file, index=False)
            
            # Save daily summary
            summary_file = os.path.join(PREDICTIONS_PATH, f"daily_summary_{target_date.strftime('%Y%m%d')}.json")
            import json
            with open(summary_file, 'w') as f:
                json.dump(daily_prediction, f, indent=2, default=str)
            
            self.logger.info(f"Predictions saved for {target_date}")
            
        except Exception as e:
            self.logger.error(f"Error generating daily predictions: {str(e)}")
    
    def save_performance_history(self):
        """Save performance history to file"""
        try:
            if len(self.performance_history) > 0:
                performance_df = pd.DataFrame(self.performance_history)
                performance_file = os.path.join(LOGS_PATH, "performance_history.csv")
                performance_df.to_csv(performance_file, index=False)
                
                # Keep only last 30 days of history
                if len(self.performance_history) > 30:
                    self.performance_history = self.performance_history[-30:]
                    
        except Exception as e:
            self.logger.error(f"Error saving performance history: {str(e)}")
    
    def get_performance_summary(self, days=7):
        """Get performance summary for last N days"""
        if len(self.performance_history) == 0:
            return None
        
        recent_performance = self.performance_history[-days:]
        
        summary = {
            'avg_accuracy': np.mean([p['accuracy_score'] for p in recent_performance]),
            'avg_sales_mape': np.mean([p['sales_mape'] for p in recent_performance]),
            'total_days_evaluated': len(recent_performance),
            'best_day': max(recent_performance, key=lambda x: x['accuracy_score']),
            'worst_day': min(recent_performance, key=lambda x: x['accuracy_score'])
        }
        
        return summary
    
    def start_continuous_learning(self):
        """Start the continuous learning scheduler"""
        self.logger.info("Starting continuous learning scheduler...")
        
        # Schedule daily learning at 2 AM
        schedule.every().day.at("02:00").do(self.daily_learning_cycle)
        
        # Schedule performance reporting at 9 AM
        schedule.every().day.at("09:00").do(self.generate_performance_report)
        
        self.logger.info("Scheduler started. Running continuous learning...")
        
        # Run initial cycle
        self.daily_learning_cycle()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def generate_performance_report(self):
        """Generate and log performance report"""
        try:
            summary = self.get_performance_summary(days=7)
            
            if summary:
                self.logger.info("=== WEEKLY PERFORMANCE REPORT ===")
                self.logger.info(f"Average Accuracy: {summary['avg_accuracy']:.2f}%")
                self.logger.info(f"Average Sales MAPE: {summary['avg_sales_mape']:.2f}%")
                self.logger.info(f"Days Evaluated: {summary['total_days_evaluated']}")
                self.logger.info(f"Best Day: {summary['best_day']['date']} ({summary['best_day']['accuracy_score']:.2f}%)")
                self.logger.info(f"Worst Day: {summary['worst_day']['date']} ({summary['worst_day']['accuracy_score']:.2f}%)")
                self.logger.info("================================")
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
    
    def manual_retrain(self, data_path=None):
        """Manually trigger model retraining"""
        self.logger.info("Manual retraining triggered...")
        
        try:
            success = self.model_trainer.train_all_models(data_path)
            
            if success:
                self.last_training_date = datetime.now().date()
                self.predictor = SalesPredictor()  # Reload with new model
                self.logger.info("Manual retraining completed successfully")
                return True
            else:
                self.logger.error("Manual retraining failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in manual retraining: {str(e)}")
            return False

def main():
    """Main function to start continuous learning"""
    learner = ContinuousLearner()
    
    # Check if we should start with initial training
    if not learner.model_trainer.load_models():
        print("No existing models found. Starting initial training...")
        success = learner.model_trainer.train_all_models()
        if not success:
            print("Initial training failed. Please check your data and try again.")
            return
    
    print("Starting continuous learning system...")
    learner.start_continuous_learning()

if __name__ == "__main__":
    main()