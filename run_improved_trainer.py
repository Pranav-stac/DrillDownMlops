#!/usr/bin/env python3
"""
Run the improved historical trainer
"""

from improved_historical_trainer import ImprovedHistoricalTrainer

def main():
    print("🚀 Starting Improved Historical Learning...")
    trainer = ImprovedHistoricalTrainer()
    trainer.simulate_historical_learning()
    print("✅ Training complete!")

if __name__ == "__main__":
    main()