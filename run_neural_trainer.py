#!/usr/bin/env python3
"""
Neural Network Sales Prediction Trainer
"""

import sys
import os
from neural_sales_trainer import NeuralSalesTrainer

def main():
    print("🧠 Starting Neural Network Sales Prediction Training...")
    
    # Check if data file exists
    if not os.path.exists("enriched_sales_data_2023_2025.csv"):
        print("❌ Error: enriched_sales_data_2023_2025.csv not found!")
        print("Please run enrich_sales_data.py first to create the enriched dataset.")
        return
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("❌ Error: TensorFlow not found!")
        print("Please install TensorFlow: pip install tensorflow")
        return
    
    # Initialize trainer
    trainer = NeuralSalesTrainer()
    
    # Data file path
    data_file = "enriched_sales_data_2023_2025.csv"
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "simulate":
        print(f"🔄 Running daily learning simulation using {data_file}...")
        success = trainer.simulate_daily_learning(data_path=data_file)
    else:
        print(f"🎓 Training neural network models using {data_file}...")
        success = trainer.train_models(data_path=data_file)
    
    if success:
        print("\n✅ Neural network training completed successfully!")
    else:
        print("\n❌ Neural network training failed!")

if __name__ == "__main__":
    main()