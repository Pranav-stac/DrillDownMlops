import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from config import *
from model_trainer import SalesPredictionModel
from feature_engineering import FeatureEngineer

class SalesPredictor:
    def __init__(self):
        self.model = SalesPredictionModel()
        self.feature_engineer = FeatureEngineer()
        
        # Setup logging
        logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
        self.logger = logging.getLogger(__name__)
        
        # Load trained models
        self.model.load_models()
        
    def get_weather_forecast(self, date, hour):
        """Get weather forecast for a specific date and hour"""
        try:
            # Format date for API
            date_str = date.strftime('%Y-%m-%d')
            
            # Open-Meteo API for forecast
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': WEATHER_LAT,
                'longitude': WEATHER_LON,
                'hourly': 'temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,pressure_msl,cloudcover',
                'start_date': date_str,
                'end_date': date_str,
                'timezone': 'Asia/Kolkata'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                hourly_data = data.get('hourly', {})
                
                if 'time' in hourly_data and len(hourly_data['time']) > hour:
                    return {
                        'temp': hourly_data['temperature_2m'][hour],
                        'humidity': hourly_data['relativehumidity_2m'][hour],
                        'precip': hourly_data['precipitation'][hour],
                        'windspeed': hourly_data['windspeed_10m'][hour],
                        'pressure': hourly_data['pressure_msl'][hour],
                        'cloudcover': hourly_data['cloudcover'][hour]
                    }
        except Exception as e:
            self.logger.warning(f"Could not fetch weather forecast: {str(e)}")
        
        # Return default weather if API fails
        return {
            'temp': 28.0,
            'humidity': 70.0,
            'precip': 0.0,
            'windspeed': 10.0,
            'pressure': 1013.0,
            'cloudcover': 50.0
        }
    
    def get_festival_info(self, date):
        """Get festival information for a specific date"""
        # Local festival database (same as in feature engineering)
        festivals = {
            # 2025 festivals
            '2025-01-14': {'name': 'Makar Sankranti', 'type': 'Hindu Festival'},
            '2025-01-26': {'name': 'Republic Day', 'type': 'National Holiday'},
            '2025-03-14': {'name': 'Holi', 'type': 'Hindu Festival'},
            '2025-03-29': {'name': 'Gudi Padwa', 'type': 'Maharashtra Festival'},
            '2025-04-14': {'name': 'Dr. Ambedkar Jayanti', 'type': 'National Holiday'},
            '2025-05-01': {'name': 'Maharashtra Day', 'type': 'State Holiday'},
            '2025-07-06': {'name': 'Ashadhi Ekadashi', 'type': 'Maharashtra Festival'},
            '2025-08-15': {'name': 'Independence Day', 'type': 'National Holiday'},
            '2025-08-28': {'name': 'Ganesh Chaturthi', 'type': 'Maharashtra Festival'},
            '2025-10-02': {'name': 'Gandhi Jayanti', 'type': 'National Holiday'},
            '2025-10-20': {'name': 'Diwali', 'type': 'Hindu Festival'},
            '2025-11-19': {'name': 'Kartiki Ekadashi', 'type': 'Maharashtra Festival'},
            
            # Add more festivals as needed
        }
        
        date_str = date.strftime('%Y-%m-%d')
        if date_str in festivals:
            festival_info = festivals[date_str]
            return {
                'name': festival_info['name'],
                'type': festival_info['type'],
                'has_festival': 1
            }
        else:
            return {
                'name': None,
                'type': None,
                'has_festival': 0
            }
    
    def create_prediction_features(self, target_date, target_hour, weather_data=None, festival_data=None):
        """Create features for prediction"""
        # Create base dataframe
        prediction_data = {
            'date': [target_date],
            'hour': [target_hour],
            'datetime': [datetime.combine(target_date, datetime.min.time().replace(hour=target_hour))]
        }
        
        # Add weather data
        if weather_data is None:
            weather_data = self.get_weather_forecast(target_date, target_hour)
        
        for key, value in weather_data.items():
            prediction_data[key] = [value]
        
        # Add festival data
        if festival_data is None:
            festival_data = self.get_festival_info(target_date)
        
        for key, value in festival_data.items():
            prediction_data[key] = [value]
        
        # Create DataFrame
        df = pd.DataFrame(prediction_data)
        
        # Add historical data for lag features (simplified approach)
        # In a real implementation, you'd fetch actual historical data
        historical_sales = self.get_historical_sales_estimate(target_date, target_hour)
        
        df['sales_lag_1h'] = [historical_sales * 0.9]
        df['sales_lag_24h'] = [historical_sales * 1.1]
        df['sales_lag_168h'] = [historical_sales * 0.95]
        df['customers_lag_1h'] = [historical_sales / 300]
        df['customers_lag_24h'] = [historical_sales / 280]
        df['customers_lag_168h'] = [historical_sales / 320]
        df['sales_rolling_mean_7h'] = [historical_sales]
        df['sales_rolling_mean_24h'] = [historical_sales]
        df['sales_rolling_mean_168h'] = [historical_sales]
        df['same_hour_last_week'] = [historical_sales * 1.05]
        df['same_hour_last_month'] = [historical_sales * 0.98]
        
        # Apply feature engineering
        df = self.feature_engineer.create_time_features(df)
        df = self.feature_engineer.create_weather_features(df)
        df = self.feature_engineer.create_festival_features(df)
        
        # Create all categorical dummy variables that were used in training
        self.add_missing_categorical_features(df)
        
        return df
    
    def add_missing_categorical_features(self, df):
        """Add missing categorical features with default values"""
        # Define all possible categorical features
        categorical_features = {
            'time_category': ['night', 'morning', 'afternoon', 'evening'],
            'temp_category': ['cold', 'cool', 'pleasant', 'warm', 'hot'],
            'humidity_category': ['dry', 'comfortable', 'humid', 'very_humid'],
            'rain_intensity': ['no_rain', 'light', 'moderate', 'heavy'],
            'wind_category': ['calm', 'light', 'moderate', 'strong']
        }
        
        for category, values in categorical_features.items():
            if category in df.columns:
                # Get current value
                current_value = df[category].iloc[0]
                
                # Create dummy variables for all possible values
                for value in values:
                    dummy_col = f"{category}_{value}"
                    df[dummy_col] = 1 if current_value == value else 0
            else:
                # If category doesn't exist, create all dummies as 0
                for value in values:
                    dummy_col = f"{category}_{value}"
                    df[dummy_col] = 0
    
    def get_historical_sales_estimate(self, target_date, target_hour):
        """Get historical sales estimate for lag features"""
        # This is a simplified approach - in production, you'd query actual historical data
        # Base estimate varies by hour and day of week
        base_sales = {
            0: 500, 1: 300, 2: 200, 3: 150, 4: 100, 5: 200,
            6: 800, 7: 1200, 8: 1500, 9: 1800, 10: 2000, 11: 2500,
            12: 3000, 13: 3200, 14: 2800, 15: 2200, 16: 1800, 17: 2000,
            18: 2500, 19: 2800, 20: 2200, 21: 1800, 22: 1200, 23: 800
        }
        
        hour_sales = base_sales.get(target_hour, 1000)
        
        # Adjust for day of week
        day_multipliers = {0: 0.9, 1: 0.85, 2: 0.9, 3: 0.95, 4: 1.1, 5: 1.3, 6: 1.2}
        day_of_week = target_date.weekday()
        
        return hour_sales * day_multipliers.get(day_of_week, 1.0)
    
    def predict_sales(self, target_date, target_hour, weather_data=None, festival_data=None):
        """Predict sales for a specific date and hour"""
        if not self.model.is_trained:
            self.logger.error("Model not trained. Please train the model first.")
            return None
        
        # Create features
        features_df = self.create_prediction_features(target_date, target_hour, weather_data, festival_data)
        
        # Make predictions
        predictions = self.model.predict(features_df)
        
        if predictions is None:
            return None
        
        # Get feature importance for reasoning
        feature_importance = self.model.get_feature_importance('hourly_sales')
        
        # Create prediction result
        result = {
            'date': target_date.strftime('%Y-%m-%d'),
            'hour': target_hour,
            'prediction_type': 'dine_in_only',
            'predictions': {
                'hourly_sales': round(predictions.get('hourly_sales', 0), 2),
                'customer_count': round(predictions.get('customer_count', 0)),
                'avg_order_value': round(predictions.get('hourly_sales', 0) / max(predictions.get('customer_count', 1), 1), 2)
            },
            'input_conditions': {
                'weather': weather_data or self.get_weather_forecast(target_date, target_hour),
                'festival': festival_data or self.get_festival_info(target_date)
            },
            'reasoning': self.generate_reasoning(features_df, feature_importance),
            'confidence': self.calculate_confidence(features_df),
            'note': 'Predictions are for dine-in customers only (online orders excluded)'
        }
        
        return result
    
    def predict_daily_sales(self, target_date, weather_data_daily=None, festival_data=None):
        """Predict sales for an entire day (24 hours)"""
        daily_predictions = []
        total_sales = 0
        total_customers = 0
        
        for hour in range(24):
            # Get hourly weather data if daily data provided
            hourly_weather = None
            if weather_data_daily and isinstance(weather_data_daily, list) and len(weather_data_daily) > hour:
                hourly_weather = weather_data_daily[hour]
            
            # Predict for this hour
            prediction = self.predict_sales(target_date, hour, hourly_weather, festival_data)
            
            if prediction:
                daily_predictions.append(prediction)
                total_sales += prediction['predictions']['hourly_sales']
                total_customers += prediction['predictions']['customer_count']
        
        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'daily_summary': {
                'total_sales': round(total_sales, 2),
                'total_customers': total_customers,
                'avg_hourly_sales': round(total_sales / 24, 2),
                'peak_hour': max(daily_predictions, key=lambda x: x['predictions']['hourly_sales'])['hour'] if daily_predictions else 12
            },
            'hourly_predictions': daily_predictions
        }
    
    def generate_reasoning(self, features_df, feature_importance):
        """Generate reasoning for predictions"""
        if feature_importance is None or len(feature_importance) == 0:
            return ["Model predictions based on historical patterns"]
        
        reasoning = []
        
        # Get top 5 most important features
        top_features = feature_importance.head(5)
        
        for _, row in top_features.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            
            if feature_name in features_df.columns:
                feature_value = features_df[feature_name].iloc[0]
                
                # Generate human-readable reasoning
                if 'temp' in feature_name:
                    if feature_value > 30:
                        reasoning.append(f"High temperature ({feature_value:.1f}°C) may increase cold beverage sales")
                    elif feature_value < 20:
                        reasoning.append(f"Cool temperature ({feature_value:.1f}°C) may increase hot beverage sales")
                    else:
                        reasoning.append(f"Pleasant temperature ({feature_value:.1f}°C) supports normal sales patterns")
                
                elif 'festival' in feature_name and feature_value > 0:
                    reasoning.append("Festival day typically increases sales by 15-25%")
                
                elif 'weekend' in feature_name and feature_value > 0:
                    reasoning.append("Weekend typically shows 20-30% higher sales")
                
                elif 'hour' in feature_name:
                    hour = features_df['hour'].iloc[0]
                    if 12 <= hour <= 14:
                        reasoning.append("Lunch hour - peak sales period")
                    elif 18 <= hour <= 20:
                        reasoning.append("Evening peak - high sales expected")
                    elif hour < 6:
                        reasoning.append("Early morning - typically low sales")
                
                elif 'rain' in feature_name and feature_value > 0:
                    reasoning.append("Rainy weather may reduce foot traffic")
        
        return reasoning[:3] if reasoning else ["Prediction based on historical sales patterns"]
    
    def calculate_confidence(self, features_df):
        """Calculate prediction confidence based on feature completeness"""
        # Simple confidence calculation based on data completeness
        total_features = len(self.model.feature_columns)
        available_features = sum(1 for col in self.model.feature_columns if col in features_df.columns)
        
        base_confidence = available_features / total_features
        
        # Adjust based on weather data availability
        weather_features = ['temp', 'humidity', 'precip', 'windspeed']
        weather_available = sum(1 for col in weather_features if col in features_df.columns)
        weather_confidence = weather_available / len(weather_features)
        
        # Combined confidence
        confidence = (base_confidence * 0.7 + weather_confidence * 0.3)
        
        return round(min(confidence, 0.95), 2)  # Cap at 95%
    
    def get_item_predictions(self, target_date, target_hour, top_n=5):
        """Predict top items for a specific date and hour"""
        # This is a simplified version - in production, you'd have item-specific models
        base_prediction = self.predict_sales(target_date, target_hour)
        
        if not base_prediction:
            return []
        
        # Popular items by hour (simplified)
        popular_items = {
            'morning': ['Coffee', 'Tea', 'Breakfast Sandwich', 'Croissant', 'Fresh Juice'],
            'lunch': ['Sandwich', 'Salad', 'Pasta', 'Pizza', 'Cold Coffee'],
            'evening': ['Tea', 'Snacks', 'Cake', 'Hot Chocolate', 'Cookies'],
            'night': ['Dessert', 'Hot Chocolate', 'Tea', 'Light Snacks', 'Ice Cream']
        }
        
        # Determine time period
        if 6 <= target_hour <= 11:
            period = 'morning'
        elif 12 <= target_hour <= 16:
            period = 'lunch'
        elif 17 <= target_hour <= 21:
            period = 'evening'
        else:
            period = 'night'
        
        items = popular_items[period][:top_n]
        
        # Estimate quantities based on total sales and customer count
        total_customers = base_prediction['predictions']['customer_count']
        
        item_predictions = []
        for i, item in enumerate(items):
            # Decrease probability for lower-ranked items
            probability = 0.9 - (i * 0.15)
            quantity = max(1, int(total_customers * probability * 0.3))
            
            item_predictions.append({
                'item': item,
                'predicted_quantity': quantity,
                'probability': round(probability, 2),
                'reasoning': f"Popular {period} item with {probability*100:.0f}% customer preference"
            })
        
        return item_predictions