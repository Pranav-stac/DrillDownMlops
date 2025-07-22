import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import joblib
from neural_sales_trainer import NeuralSalesTrainer
from improved_historical_trainer import ImprovedHistoricalTrainer

class SalesDashboard:
    def __init__(self):
        self.neural_trainer = None
        self.xgb_trainer = None
        self.performance_data = None
        self.load_performance_data()
        
    def load_performance_data(self):
        """Load performance data from CSV files"""
        files = [
            'neural_daily_learning_performance.csv',
            'hybrid_daily_learning_performance.csv',
            'improved_daily_learning_performance.csv',
            'improved_daily_learning_performance_2024_2025.csv'
        ]
        
        for file in files:
            if os.path.exists(file):
                self.performance_data = pd.read_csv(file)
                print(f"Loaded performance data from {file}")
                self.performance_data['date'] = pd.to_datetime(self.performance_data['date'])
                break
        
        if self.performance_data is None:
            print("No performance data found. Please run training first.")
    
    def load_models(self):
        """Load trained models"""
        try:
            # Try to load neural network models
            self.neural_trainer = NeuralSalesTrainer()
            if self.neural_trainer.load_models():
                print("Neural network models loaded successfully")
        except Exception as e:
            print(f"Could not load neural network models: {str(e)}")
        
        try:
            # Try to load XGBoost models
            self.xgb_trainer = ImprovedHistoricalTrainer()
            if hasattr(self.xgb_trainer, 'load_models') and self.xgb_trainer.load_models():
                print("XGBoost models loaded successfully")
        except Exception as e:
            print(f"Could not load XGBoost models: {str(e)}")
    
    def show_performance_trends(self):
        """Show performance trends over time"""
        if self.performance_data is None:
            print("No performance data available")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot sales accuracy
        plt.subplot(2, 1, 1)
        plt.plot(self.performance_data['date'], self.performance_data['sales_accuracy'], 'b-', label='Sales Accuracy')
        plt.title('Sales Prediction Accuracy Over Time')
        plt.xlabel('Date')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()
        
        # Plot customer accuracy
        plt.subplot(2, 1, 2)
        plt.plot(self.performance_data['date'], self.performance_data['customer_accuracy'], 'g-', label='Customer Count Accuracy')
        plt.title('Customer Count Prediction Accuracy Over Time')
        plt.xlabel('Date')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('performance_trends.png')
        plt.show()
        
        print(f"Performance trends saved to performance_trends.png")
    
    def show_actual_vs_predicted(self):
        """Show actual vs predicted sales"""
        if self.performance_data is None:
            print("No performance data available")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot actual vs predicted sales
        plt.subplot(2, 1, 1)
        plt.plot(self.performance_data['date'], self.performance_data['actual_sales'], 'b-', label='Actual Sales')
        plt.plot(self.performance_data['date'], self.performance_data['predicted_sales'], 'r--', label='Predicted Sales')
        plt.title('Actual vs Predicted Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales Amount')
        plt.grid(True)
        plt.legend()
        
        # Plot actual vs predicted customers
        plt.subplot(2, 1, 2)
        plt.plot(self.performance_data['date'], self.performance_data['actual_customers'], 'g-', label='Actual Customers')
        plt.plot(self.performance_data['date'], self.performance_data['predicted_customers'], 'm--', label='Predicted Customers')
        plt.title('Actual vs Predicted Customer Count')
        plt.xlabel('Date')
        plt.ylabel('Customer Count')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')
        plt.show()
        
        print(f"Actual vs predicted comparison saved to actual_vs_predicted.png")
    
    def predict_future(self, days=7):
        """Predict future sales for the next few days"""
        if self.neural_trainer is None and self.xgb_trainer is None:
            print("No models loaded. Loading models...")
            self.load_models()
            
            if self.neural_trainer is None and self.xgb_trainer is None:
                print("Could not load any models. Please train models first.")
                return
        
        # Get last date from performance data
        if self.performance_data is not None:
            last_date = self.performance_data['date'].max().date()
        else:
            last_date = datetime.now().date()
        
        # Predict for future days
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        print(f"\n📊 Sales Forecast for Next {days} Days:")
        print("="*60)
        
        for future_date in future_dates:
            # Use neural trainer if available, otherwise use XGBoost
            if self.neural_trainer is not None and self.neural_trainer.is_trained:
                # This is a simplified version - in a real implementation, you'd need to:
                # 1. Get weather forecast for the future date
                # 2. Check if it's a festival day
                # 3. Create proper features for prediction
                print(f"Date: {future_date.strftime('%Y-%m-%d')} (Neural prediction not implemented in this demo)")
            elif self.xgb_trainer is not None and hasattr(self.xgb_trainer, 'is_trained') and self.xgb_trainer.is_trained:
                print(f"Date: {future_date.strftime('%Y-%m-%d')} (XGBoost prediction not implemented in this demo)")
            else:
                print(f"Date: {future_date.strftime('%Y-%m-%d')} (No trained models available)")
        
        print("\n⚠️ Note: This is a simplified dashboard. For actual predictions,")
        print("   you would need to implement the full prediction logic using")
        print("   the trained models and proper feature engineering.")

def main():
    print("\n" + "="*60)
    print("📊 SALES PREDICTION DASHBOARD")
    print("="*60)
    
    dashboard = SalesDashboard()
    
    while True:
        print("\nOptions:")
        print("1. Show Performance Trends")
        print("2. Show Actual vs Predicted")
        print("3. Predict Future Sales")
        print("4. Load Models")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            dashboard.show_performance_trends()
        elif choice == '2':
            dashboard.show_actual_vs_predicted()
        elif choice == '3':
            days = int(input("Enter number of days to predict: "))
            dashboard.predict_future(days)
        elif choice == '4':
            dashboard.load_models()
        elif choice == '5':
            print("Exiting dashboard...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()