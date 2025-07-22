import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        self.target_columns = ['hourly_sales', 'customer_count']
        
    def create_time_features(self, df):
        """Create time-based features"""
        df = df.copy()
        
        # Convert date to datetime if not already
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')
        
        # Basic time features
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['quarter'] = df['datetime'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['datetime'].dt.day <= 7).astype(int)
        df['is_month_end'] = (df['datetime'].dt.day >= 24).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time of day categories
        df['time_category'] = pd.cut(df['hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['night', 'morning', 'afternoon', 'evening'],
                                   include_lowest=True)
        
        return df
    
    def create_weather_features(self, df):
        """Create weather-based features"""
        df = df.copy()
        
        # Weather categories
        df['temp_category'] = pd.cut(df['temp'], 
                                   bins=[0, 20, 25, 30, 35, 50], 
                                   labels=['cold', 'cool', 'pleasant', 'warm', 'hot'])
        
        df['humidity_category'] = pd.cut(df['humidity'], 
                                       bins=[0, 40, 60, 80, 100], 
                                       labels=['dry', 'comfortable', 'humid', 'very_humid'])
        
        # Weather comfort score (combination of temp and humidity)
        df['comfort_score'] = 100 - abs(df['temp'] - 25) * 2 - abs(df['humidity'] - 50) * 0.5
        df['comfort_score'] = df['comfort_score'].clip(0, 100)
        
        # Rain indicator
        df['is_raining'] = (df['precip'] > 0).astype(int)
        df['rain_intensity'] = pd.cut(df['precip'], 
                                    bins=[0, 0.1, 2.5, 10, 50], 
                                    labels=['no_rain', 'light', 'moderate', 'heavy'])
        
        # Wind categories
        df['wind_category'] = pd.cut(df['windspeed'], 
                                   bins=[0, 10, 20, 30, 100], 
                                   labels=['calm', 'light', 'moderate', 'strong'])
        
        return df
    
    def create_festival_features(self, df):
        """Create festival-based features"""
        df = df.copy()
        
        # Festival impact features
        df['has_festival'] = df['has_festival'].fillna(0).astype(int)
        
        # Festival type encoding
        if 'type' in df.columns:
            festival_types = ['Hindu Festival', 'National Holiday', 'Maharashtra Festival', 
                            'State Holiday', 'Fasting Day']
            for fest_type in festival_types:
                df[f'festival_{fest_type.lower().replace(" ", "_")}'] = (
                    df['type'] == fest_type).astype(int)
        
        # Days to/from festival
        festival_dates = df[df['has_festival'] == 1]['datetime'].unique()
        if len(festival_dates) > 0:
            df['days_to_next_festival'] = df['datetime'].apply(
                lambda x: min([abs((fest - x).days) for fest in festival_dates] + [365])
            )
            df['is_festival_week'] = (df['days_to_next_festival'] <= 7).astype(int)
        else:
            df['days_to_next_festival'] = 365
            df['is_festival_week'] = 0
        
        return df
    
    def create_lag_features(self, df, target_col='final_total'):
        """Create lag and rolling features"""
        df = df.copy()
        df = df.sort_values('datetime')
        
        # Filter only dine-in orders (exclude online orders)
        if 'is_dine_in' in df.columns:
            df = df[df['is_dine_in'] == True]
            print(f"Filtered to {len(df)} dine-in records only")
        
        # Aggregate to hourly level first
        hourly_df = df.groupby(['datetime']).agg({
            'final_total': 'sum',
            'quantity': 'sum',
            'item_name': 'count'  # customer count proxy
        }).reset_index()
        
        hourly_df.columns = ['datetime', 'hourly_sales', 'hourly_quantity', 'customer_count']
        
        # Create lag features
        for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
            hourly_df[f'sales_lag_{lag}h'] = hourly_df['hourly_sales'].shift(lag)
            hourly_df[f'customers_lag_{lag}h'] = hourly_df['customer_count'].shift(lag)
        
        # Rolling features
        for window in [7, 24, 168]:  # 7 hours, 1 day, 1 week
            hourly_df[f'sales_rolling_mean_{window}h'] = (
                hourly_df['hourly_sales'].rolling(window=window, min_periods=1).mean()
            )
            hourly_df[f'sales_rolling_std_{window}h'] = (
                hourly_df['hourly_sales'].rolling(window=window, min_periods=1).std()
            )
        
        # Same hour last week/month
        hourly_df['same_hour_last_week'] = hourly_df['hourly_sales'].shift(168)
        hourly_df['same_hour_last_month'] = hourly_df['hourly_sales'].shift(168 * 4)
        
        return hourly_df
    
    def create_item_features(self, df):
        """Create item-based features"""
        df = df.copy()
        
        # Item popularity score
        item_counts = df.groupby('item_name')['quantity'].sum()
        df['item_popularity'] = df['item_name'].map(item_counts)
        
        # Price categories
        df['price_category'] = pd.cut(df['price'], 
                                    bins=[0, 200, 500, 1000, 5000], 
                                    labels=['budget', 'mid', 'premium', 'luxury'])
        
        # Order type features
        df['is_dine_in'] = df['is_dine_in'].astype(int)
        df['is_online'] = df['is_online'].astype(int)
        
        return df
    
    def prepare_features(self, df):
        """Main function to prepare all features"""
        print(f"Starting with {len(df)} total records")
        
        # Filter only dine-in orders at the beginning
        if 'is_dine_in' in df.columns:
            df = df[df['is_dine_in'] == True]
            print(f"Filtered to {len(df)} dine-in records only (excluding online orders)")
        
        print("Creating time features...")
        df = self.create_time_features(df)
        
        print("Creating weather features...")
        df = self.create_weather_features(df)
        
        print("Creating festival features...")
        df = self.create_festival_features(df)
        
        print("Creating item features...")
        df = self.create_item_features(df)
        
        print("Creating lag features...")
        hourly_df = self.create_lag_features(df)
        
        # Merge back with original data
        df_with_datetime = df.copy()
        if 'datetime' not in df_with_datetime.columns:
            df_with_datetime['datetime'] = pd.to_datetime(
                df_with_datetime['date'].astype(str) + ' ' + 
                df_with_datetime['hour'].astype(str) + ':00:00'
            )
        
        # Get unique datetime records for merging
        unique_datetime_features = df_with_datetime.groupby('datetime').first().reset_index()
        
        # Merge with lag features
        final_df = pd.merge(hourly_df, unique_datetime_features, on='datetime', how='left')
        
        # Select feature columns
        feature_cols = [
            # Time features
            'hour', 'day_of_week', 'month', 'quarter', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            
            # Weather features
            'temp', 'humidity', 'precip', 'windspeed', 'pressure', 'cloudcover',
            'comfort_score', 'is_raining',
            
            # Festival features
            'has_festival', 'days_to_next_festival', 'is_festival_week',
            
            # Lag features
            'sales_lag_1h', 'sales_lag_24h', 'sales_lag_168h',
            'customers_lag_1h', 'customers_lag_24h', 'customers_lag_168h',
            'sales_rolling_mean_7h', 'sales_rolling_mean_24h', 'sales_rolling_mean_168h',
            'same_hour_last_week', 'same_hour_last_month'
        ]
        
        # Add categorical features if they exist
        categorical_features = []
        for col in ['time_category', 'temp_category', 'humidity_category', 'rain_intensity', 'wind_category']:
            if col in final_df.columns:
                # One-hot encode categorical features
                dummies = pd.get_dummies(final_df[col], prefix=col)
                final_df = pd.concat([final_df, dummies], axis=1)
                categorical_features.extend(dummies.columns.tolist())
        
        feature_cols.extend(categorical_features)
        
        # Filter existing columns
        existing_feature_cols = [col for col in feature_cols if col in final_df.columns]
        self.feature_columns = existing_feature_cols
        
        # Fill missing values
        for col in existing_feature_cols:
            if final_df[col].dtype in ['float64', 'int64']:
                final_df[col] = final_df[col].fillna(final_df[col].median())
            else:
                final_df[col] = final_df[col].fillna(0)
        
        print(f"Created {len(existing_feature_cols)} features")
        return final_df[['datetime'] + existing_feature_cols + ['hourly_sales', 'customer_count']]
    
    def get_feature_importance_names(self):
        """Get feature names for importance analysis"""
        return self.feature_columns