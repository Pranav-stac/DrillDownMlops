import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import json
import os
from predictor import SalesPredictor
from model_trainer import SalesPredictionModel
from continuous_learner import ContinuousLearner
from config import *

# Page configuration
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SalesDashboard:
    def __init__(self):
        self.predictor = SalesPredictor()
        self.model_trainer = SalesPredictionModel()
        self.continuous_learner = ContinuousLearner()
        
        # Load models if available
        if not self.predictor.model.is_trained:
            self.predictor.model.load_models()
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🎯 Sales Prediction Controls")
        
        # Date and time selection
        st.sidebar.subheader("📅 Prediction Settings")
        
        prediction_date = st.sidebar.date_input(
            "Select Date",
            value=datetime.now().date() + timedelta(days=1),
            min_value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=30)
        )
        
        prediction_hour = st.sidebar.selectbox(
            "Select Hour",
            options=list(range(24)),
            index=12,
            format_func=lambda x: f"{x:02d}:00"
        )
        
        # Weather inputs
        st.sidebar.subheader("🌤️ Weather Conditions")
        
        use_forecast = st.sidebar.checkbox("Use Weather Forecast", value=True)
        
        if not use_forecast:
            temperature = st.sidebar.slider("Temperature (°C)", 15, 45, 28)
            humidity = st.sidebar.slider("Humidity (%)", 20, 100, 70)
            precipitation = st.sidebar.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
            wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 10)
            cloud_cover = st.sidebar.slider("Cloud Cover (%)", 0, 100, 50)
            
            weather_data = {
                'temp': temperature,
                'humidity': humidity,
                'precip': precipitation,
                'windspeed': wind_speed,
                'pressure': 1013.0,
                'cloudcover': cloud_cover
            }
        else:
            weather_data = None
        
        # Festival information
        st.sidebar.subheader("🎉 Festival Information")
        
        has_festival = st.sidebar.checkbox("Is Festival Day?", value=False)
        
        if has_festival:
            festival_name = st.sidebar.text_input("Festival Name", "Custom Festival")
            festival_type = st.sidebar.selectbox(
                "Festival Type",
                ["Hindu Festival", "National Holiday", "Maharashtra Festival", "State Holiday", "Fasting Day"]
            )
            
            festival_data = {
                'name': festival_name,
                'type': festival_type,
                'has_festival': 1
            }
        else:
            festival_data = None
        
        return prediction_date, prediction_hour, weather_data, festival_data
    
    def render_main_prediction(self, prediction_date, prediction_hour, weather_data, festival_data):
        """Render main prediction section"""
        st.title("📊 Sales Prediction Dashboard")
        
        if not self.predictor.model.is_trained:
            st.error("⚠️ Model not trained yet. Please train the model first using the Training section.")
            return
        
        # Make prediction
        with st.spinner("🔮 Generating predictions..."):
            prediction = self.predictor.predict_sales(
                prediction_date, prediction_hour, weather_data, festival_data
            )
        
        if prediction is None:
            st.error("❌ Failed to generate prediction. Please check your inputs.")
            return
        
        # Display main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Expected Sales",
                f"₹{prediction['predictions']['hourly_sales']:,.0f}",
                help="Predicted sales amount for the selected hour"
            )
        
        with col2:
            st.metric(
                "👥 Expected Customers",
                f"{prediction['predictions']['customer_count']:,}",
                help="Predicted number of customers for the selected hour"
            )
        
        with col3:
            st.metric(
                "🛒 Avg Order Value",
                f"₹{prediction['predictions']['avg_order_value']:.0f}",
                help="Average order value per customer"
            )
        
        with col4:
            confidence_color = "green" if prediction['confidence'] > 0.8 else "orange" if prediction['confidence'] > 0.6 else "red"
            st.metric(
                "🎯 Confidence",
                f"{prediction['confidence']*100:.0f}%",
                help="Prediction confidence level"
            )
        
        # Reasoning section
        st.subheader("🧠 Prediction Reasoning")
        
        reasoning_col1, reasoning_col2 = st.columns(2)
        
        with reasoning_col1:
            st.write("**Key Factors:**")
            for reason in prediction['reasoning']:
                st.write(f"• {reason}")
        
        with reasoning_col2:
            st.write("**Input Conditions:**")
            
            # Weather conditions
            weather = prediction['input_conditions']['weather']
            st.write(f"🌡️ **Temperature:** {weather['temp']:.1f}°C")
            st.write(f"💧 **Humidity:** {weather['humidity']:.0f}%")
            if weather['precip'] > 0:
                st.write(f"🌧️ **Precipitation:** {weather['precip']:.1f}mm")
            st.write(f"💨 **Wind Speed:** {weather['windspeed']:.1f} km/h")
            
            # Festival information
            festival = prediction['input_conditions']['festival']
            if festival['has_festival']:
                st.write(f"🎉 **Festival:** {festival['name']} ({festival['type']})")
            else:
                st.write("🎉 **Festival:** No festival")
    
    def render_daily_prediction(self, prediction_date, weather_data, festival_data):
        """Render daily prediction section"""
        st.subheader("📈 Daily Sales Forecast")
        
        if not self.predictor.model.is_trained:
            st.warning("Model not trained yet.")
            return
        
        with st.spinner("Generating daily forecast..."):
            daily_prediction = self.predictor.predict_daily_sales(
                prediction_date, None, festival_data
            )
        
        if daily_prediction is None:
            st.error("Failed to generate daily prediction.")
            return
        
        # Daily summary
        summary = daily_prediction['daily_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Total Daily Sales", f"₹{summary['total_sales']:,.0f}")
        
        with col2:
            st.metric("👥 Total Customers", f"{summary['total_customers']:,}")
        
        with col3:
            st.metric("📊 Avg Hourly Sales", f"₹{summary['avg_hourly_sales']:,.0f}")
        
        with col4:
            st.metric("⏰ Peak Hour", f"{summary['peak_hour']:02d}:00")
        
        # Hourly chart
        hourly_data = []
        for hour_pred in daily_prediction['hourly_predictions']:
            hourly_data.append({
                'Hour': f"{hour_pred['hour']:02d}:00",
                'Sales': hour_pred['predictions']['hourly_sales'],
                'Customers': hour_pred['predictions']['customer_count'],
                'Confidence': hour_pred['confidence']
            })
        
        hourly_df = pd.DataFrame(hourly_data)
        
        # Sales chart
        fig_sales = px.line(
            hourly_df, x='Hour', y='Sales',
            title='Hourly Sales Forecast',
            labels={'Sales': 'Sales (₹)', 'Hour': 'Time'},
            line_shape='spline'
        )
        fig_sales.update_layout(height=400)
        st.plotly_chart(fig_sales, use_container_width=True)
        
        # Customer chart
        fig_customers = px.bar(
            hourly_df, x='Hour', y='Customers',
            title='Hourly Customer Forecast',
            labels={'Customers': 'Number of Customers', 'Hour': 'Time'}
        )
        fig_customers.update_layout(height=400)
        st.plotly_chart(fig_customers, use_container_width=True)
    
    def render_item_predictions(self, prediction_date, prediction_hour):
        """Render item prediction section"""
        st.subheader("🍽️ Popular Items Forecast")
        
        if not self.predictor.model.is_trained:
            st.warning("Model not trained yet.")
            return
        
        with st.spinner("Predicting popular items..."):
            item_predictions = self.predictor.get_item_predictions(
                prediction_date, prediction_hour, top_n=10
            )
        
        if not item_predictions:
            st.warning("No item predictions available.")
            return
        
        # Create DataFrame for display
        items_df = pd.DataFrame(item_predictions)
        
        # Display as table
        st.dataframe(
            items_df[['item', 'predicted_quantity', 'probability', 'reasoning']],
            column_config={
                'item': 'Item Name',
                'predicted_quantity': st.column_config.NumberColumn('Predicted Quantity', format='%d'),
                'probability': st.column_config.NumberColumn('Probability', format='%.0%%'),
                'reasoning': 'Reasoning'
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Chart
        fig_items = px.bar(
            items_df.head(5), x='predicted_quantity', y='item',
            title='Top 5 Items - Predicted Quantities',
            labels={'predicted_quantity': 'Predicted Quantity', 'item': 'Item'},
            orientation='h'
        )
        fig_items.update_layout(height=400)
        st.plotly_chart(fig_items, use_container_width=True)
    
    def render_model_performance(self):
        """Render model performance section"""
        st.subheader("📊 Model Performance")
        
        # Load performance history
        performance_file = os.path.join(LOGS_PATH, "performance_history.csv")
        
        if os.path.exists(performance_file):
            performance_df = pd.read_csv(performance_file)
            performance_df['date'] = pd.to_datetime(performance_df['date'])
            
            if len(performance_df) > 0:
                # Performance metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_accuracy = performance_df['accuracy_score'].mean()
                    st.metric("📈 Average Accuracy", f"{avg_accuracy:.1f}%")
                
                with col2:
                    avg_mape = performance_df['sales_mape'].mean()
                    st.metric("📉 Average MAPE", f"{avg_mape:.1f}%")
                
                with col3:
                    days_tracked = len(performance_df)
                    st.metric("📅 Days Tracked", f"{days_tracked}")
                
                # Performance chart
                fig_perf = px.line(
                    performance_df, x='date', y='accuracy_score',
                    title='Model Accuracy Over Time',
                    labels={'accuracy_score': 'Accuracy (%)', 'date': 'Date'}
                )
                fig_perf.update_layout(height=400)
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Recent performance table
                st.write("**Recent Performance:**")
                recent_df = performance_df.tail(7)[['date', 'accuracy_score', 'sales_mape', 'total_actual_sales', 'total_predicted_sales']]
                st.dataframe(recent_df, use_container_width=True)
            else:
                st.info("No performance data available yet.")
        else:
            st.info("No performance history found. The model will start tracking performance after the first predictions.")
    
    def render_training_section(self):
        """Render model training section"""
        st.subheader("🎓 Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Model Status:**")
            if self.predictor.model.is_trained:
                st.success("✅ Model is trained and ready")
                
                # Show feature importance
                feature_importance = self.predictor.model.get_feature_importance()
                if feature_importance is not None:
                    st.write("**Top 10 Important Features:**")
                    top_features = feature_importance.head(10)
                    
                    fig_importance = px.bar(
                        top_features, x='importance', y='feature',
                        title='Feature Importance',
                        orientation='h'
                    )
                    fig_importance.update_layout(height=400)
                    st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.warning("⚠️ Model not trained yet")
        
        with col2:
            st.write("**Training Controls:**")
            
            if st.button("🚀 Train New Model", type="primary"):
                with st.spinner("Training model... This may take a few minutes."):
                    success = self.model_trainer.train_all_models()
                    
                    if success:
                        st.success("✅ Model trained successfully!")
                        # Reload predictor
                        self.predictor = SalesPredictor()
                        st.rerun()
                    else:
                        st.error("❌ Model training failed. Please check the logs.")
            
            if st.button("🔄 Incremental Training"):
                with st.spinner("Performing incremental training..."):
                    success = self.continuous_learner.manual_retrain()
                    
                    if success:
                        st.success("✅ Incremental training completed!")
                        self.predictor = SalesPredictor()
                        st.rerun()
                    else:
                        st.error("❌ Incremental training failed.")
            
            # Data info
            if os.path.exists(DATA_PATH):
                data_df = pd.read_csv(DATA_PATH)
                st.info(f"📊 Training data: {len(data_df):,} records")
                
                data_df['date'] = pd.to_datetime(data_df['date'])
                date_range = f"{data_df['date'].min().date()} to {data_df['date'].max().date()}"
                st.info(f"📅 Date range: {date_range}")
    
    def run(self):
        """Main dashboard function"""
        # Sidebar
        prediction_date, prediction_hour, weather_data, festival_data = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Hourly Prediction", 
            "📈 Daily Forecast", 
            "🍽️ Item Predictions", 
            "📊 Performance", 
            "🎓 Training"
        ])
        
        with tab1:
            self.render_main_prediction(prediction_date, prediction_hour, weather_data, festival_data)
        
        with tab2:
            self.render_daily_prediction(prediction_date, weather_data, festival_data)
        
        with tab3:
            self.render_item_predictions(prediction_date, prediction_hour)
        
        with tab4:
            self.render_model_performance()
        
        with tab5:
            self.render_training_section()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray;'>
                Sales Prediction Dashboard | Powered by XGBoost & Streamlit
            </div>
            """, 
            unsafe_allow_html=True
        )

def main():
    """Main function to run the dashboard"""
    dashboard = SalesDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()