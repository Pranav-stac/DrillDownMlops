#!/usr/bin/env python3
"""
Historical Learning Script
Simulates day-by-day learning process with historical data
"""

import sys
import os
from historical_learning_trainer import HistoricalLearningTrainer

def main():
    print("🚀 Starting Historical Continuous Learning...")
    
    # Check if data file exists
    if not os.path.exists("enriched_sales_data_2023_2025.csv"):
        print("❌ Error: enriched_sales_data_2023_2025.csv not found!")
        print("Please run enrich_sales_data.py first to create the enriched dataset.")
        return
    
    # Initialize trainer
    trainer = HistoricalLearningTrainer()
    
    # Run historical learning simulation
    success = trainer.simulate_historical_learning()
    
    if success:
        print("\n✅ Historical learning completed successfully!")
        print("📊 Check the generated CSV files for detailed results:")
        print("   - daily_learning_performance.csv")
        print("   - learning_progress.csv")
    else:
        print("\n❌ Historical learning failed!")

if __name__ == "__main__":
    main()