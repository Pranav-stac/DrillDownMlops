#!/usr/bin/env python3
"""
Sales Prediction System - Main Entry Point

This script provides a command-line interface to run different components
of the sales prediction system.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_trainer import SalesPredictionModel
from predictor import SalesPredictor
from continuous_learner import ContinuousLearner
from config import *

def train_model(args):
    """Train the prediction model"""
    print("🎓 Starting model training...")
    
    trainer = SalesPredictionModel()
    
    # Determine data path
    data_path = args.data_path if args.data_path else DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure your data file exists or run data enrichment first.")
        return False
    
    # Train model
    success = trainer.train_all_models(data_path)
    
    if success:
        print("✅ Model training completed successfully!")
        
        # Show evaluation results
        print("\n📊 Evaluating model performance...")
        results = trainer.evaluate_model(data_path)
        
        if results:
            for target, metrics in results.items():
                print(f"\n{target.upper()} MODEL:")
                print(f"  MAE: {metrics['mae']:.2f}")
                print(f"  RMSE: {metrics['rmse']:.2f}")
                print(f"  R²: {metrics['r2']:.3f}")
                print(f"  Samples: {metrics['samples']}")
        
        return True
    else:
        print("❌ Model training failed!")
        return False

def make_prediction(args):
    """Make a single prediction"""
    print("🔮 Making sales prediction...")
    
    predictor = SalesPredictor()
    
    if not predictor.model.is_trained:
        print("❌ Model not trained yet. Please train the model first.")
        return False
    
    # Parse date and hour
    try:
        if args.date:
            prediction_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        else:
            prediction_date = datetime.now().date() + timedelta(days=1)
        
        prediction_hour = args.hour if args.hour is not None else 12
        
    except ValueError:
        print("❌ Invalid date format. Please use YYYY-MM-DD format.")
        return False
    
    # Make prediction
    result = predictor.predict_sales(prediction_date, prediction_hour)
    
    if result:
        print(f"\n📊 PREDICTION FOR {prediction_date} at {prediction_hour:02d}:00")
        print("=" * 50)
        print(f"💰 Expected Sales: ₹{result['predictions']['hourly_sales']:,.0f}")
        print(f"👥 Expected Customers: {result['predictions']['customer_count']:,}")
        print(f"🛒 Avg Order Value: ₹{result['predictions']['avg_order_value']:.0f}")
        print(f"🎯 Confidence: {result['confidence']*100:.0f}%")
        
        print(f"\n🧠 REASONING:")
        for reason in result['reasoning']:
            print(f"  • {reason}")
        
        print(f"\n🌤️ WEATHER CONDITIONS:")
        weather = result['input_conditions']['weather']
        print(f"  Temperature: {weather['temp']:.1f}°C")
        print(f"  Humidity: {weather['humidity']:.0f}%")
        print(f"  Precipitation: {weather['precip']:.1f}mm")
        print(f"  Wind Speed: {weather['windspeed']:.1f} km/h")
        
        festival = result['input_conditions']['festival']
        if festival['has_festival']:
            print(f"\n🎉 FESTIVAL: {festival['name']} ({festival['type']})")
        
        return True
    else:
        print("❌ Failed to generate prediction.")
        return False

def daily_forecast(args):
    """Generate daily forecast"""
    print("📈 Generating daily sales forecast...")
    
    predictor = SalesPredictor()
    
    if not predictor.model.is_trained:
        print("❌ Model not trained yet. Please train the model first.")
        return False
    
    # Parse date
    try:
        if args.date:
            prediction_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        else:
            prediction_date = datetime.now().date() + timedelta(days=1)
    except ValueError:
        print("❌ Invalid date format. Please use YYYY-MM-DD format.")
        return False
    
    # Generate daily forecast
    daily_prediction = predictor.predict_daily_sales(prediction_date)
    
    if daily_prediction:
        summary = daily_prediction['daily_summary']
        
        print(f"\n📊 DAILY FORECAST FOR {prediction_date}")
        print("=" * 50)
        print(f"💰 Total Daily Sales: ₹{summary['total_sales']:,.0f}")
        print(f"👥 Total Customers: {summary['total_customers']:,}")
        print(f"📊 Avg Hourly Sales: ₹{summary['avg_hourly_sales']:,.0f}")
        print(f"⏰ Peak Hour: {summary['peak_hour']:02d}:00")
        
        if args.detailed:
            print(f"\n⏰ HOURLY BREAKDOWN:")
            print("-" * 40)
            for hour_pred in daily_prediction['hourly_predictions']:
                hour = hour_pred['hour']
                sales = hour_pred['predictions']['hourly_sales']
                customers = hour_pred['predictions']['customer_count']
                confidence = hour_pred['confidence']
                
                print(f"{hour:02d}:00 | ₹{sales:6.0f} | {customers:2.0f} customers | {confidence*100:2.0f}% confidence")
        
        return True
    else:
        print("❌ Failed to generate daily forecast.")
        return False

def start_continuous_learning(args):
    """Start continuous learning system"""
    print("🔄 Starting continuous learning system...")
    
    learner = ContinuousLearner()
    
    # Check if models exist
    if not learner.model_trainer.load_models():
        print("No existing models found. Starting initial training...")
        success = learner.model_trainer.train_all_models()
        if not success:
            print("❌ Initial training failed. Please check your data and try again.")
            return False
    
    print("✅ Models loaded successfully.")
    print("🚀 Starting continuous learning scheduler...")
    print("The system will:")
    print("  • Fetch new data daily at 2:00 AM")
    print("  • Evaluate previous predictions")
    print("  • Retrain models when needed")
    print("  • Generate daily performance reports")
    print("\nPress Ctrl+C to stop the system.")
    
    try:
        learner.start_continuous_learning()
    except KeyboardInterrupt:
        print("\n🛑 Continuous learning system stopped.")
        return True

def run_dashboard(args):
    """Run the Streamlit dashboard"""
    print("🚀 Starting Sales Prediction Dashboard...")
    print("Dashboard will open in your web browser.")
    print("Press Ctrl+C to stop the dashboard.")
    
    import subprocess
    import sys
    
    try:
        # Run streamlit dashboard
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped.")
        return True

def show_status(args):
    """Show system status"""
    print("📊 SALES PREDICTION SYSTEM STATUS")
    print("=" * 40)
    
    # Check model status
    trainer = SalesPredictionModel()
    model_loaded = trainer.load_models()
    
    if model_loaded:
        print("✅ Models: Trained and loaded")
        print(f"   Available models: {', '.join(trainer.models.keys())}")
    else:
        print("❌ Models: Not trained")
    
    # Check data status
    if os.path.exists(DATA_PATH):
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Data: {len(df):,} records available")
        
        df['date'] = pd.to_datetime(df['date'])
        print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    else:
        print("❌ Data: No data file found")
    
    # Check performance history
    performance_file = os.path.join(LOGS_PATH, "performance_history.csv")
    if os.path.exists(performance_file):
        import pandas as pd
        perf_df = pd.read_csv(performance_file)
        print(f"✅ Performance: {len(perf_df)} days tracked")
        
        if len(perf_df) > 0:
            avg_accuracy = perf_df['accuracy_score'].mean()
            print(f"   Average accuracy: {avg_accuracy:.1f}%")
    else:
        print("❌ Performance: No history available")
    
    # Check predictions
    prediction_files = [f for f in os.listdir(PREDICTIONS_PATH) if f.startswith('predictions_')]
    if prediction_files:
        print(f"✅ Predictions: {len(prediction_files)} days of predictions available")
    else:
        print("❌ Predictions: No predictions generated yet")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Sales Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                          # Train the model
  python main.py predict --date 2025-07-25     # Predict for specific date
  python main.py forecast --date 2025-07-25    # Daily forecast
  python main.py dashboard                      # Run web dashboard
  python main.py continuous                     # Start continuous learning
  python main.py status                         # Show system status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the prediction model')
    train_parser.add_argument('--data-path', help='Path to training data CSV file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make a single prediction')
    predict_parser.add_argument('--date', help='Prediction date (YYYY-MM-DD)')
    predict_parser.add_argument('--hour', type=int, help='Prediction hour (0-23)')
    
    # Forecast command
    forecast_parser = subparsers.add_parser('forecast', help='Generate daily forecast')
    forecast_parser.add_argument('--date', help='Forecast date (YYYY-MM-DD)')
    forecast_parser.add_argument('--detailed', action='store_true', help='Show hourly breakdown')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run web dashboard')
    
    # Continuous learning command
    continuous_parser = subparsers.add_parser('continuous', help='Start continuous learning')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    commands = {
        'train': train_model,
        'predict': make_prediction,
        'forecast': daily_forecast,
        'dashboard': run_dashboard,
        'continuous': start_continuous_learning,
        'status': show_status
    }
    
    if args.command in commands:
        success = commands[args.command](args)
        sys.exit(0 if success else 1)
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()