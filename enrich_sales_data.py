import pandas as pd
import datetime
import os
import requests
import time
from datetime import timedelta
from google.cloud import bigquery
from google.oauth2 import service_account

def get_weather_data(latitude, longitude, start_date, end_date):
    """
    Fetch weather data from Open-Meteo API (free, no API key required)
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        start_date (str): Start date in format YYYY-MM-DD
        end_date (str): End date in format YYYY-MM-DD
        
    Returns:
        pandas.DataFrame: DataFrame containing hourly weather data
    """
    # Open-Meteo API has a limit of ~2 months of hourly data per request
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'temperature_2m,relativehumidity_2m,precipitation,windspeed_10m,pressure_msl,cloudcover',
        'timezone': 'Asia/Kolkata'
    }
    
    print(f"Fetching weather data from {start_date} to {end_date}...")
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    
    data = response.json()
    
    # Process the data into a DataFrame
    hourly_data = data.get('hourly', {})
    
    if not hourly_data or 'time' not in hourly_data:
        print("No hourly data found in the response")
        return None
    
    df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly_data.get('time')),
        'temp': hourly_data.get('temperature_2m'),
        'humidity': hourly_data.get('relativehumidity_2m'),
        'precip': hourly_data.get('precipitation'),
        'windspeed': hourly_data.get('windspeed_10m'),
        'pressure': hourly_data.get('pressure_msl'),
        'cloudcover': hourly_data.get('cloudcover')
    })
    
    # Add date and hour columns
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    
    return df

def get_data_in_chunks(latitude, longitude, start_date, end_date, chunk_days=60):
    """
    Get data in chunks to avoid API limitations
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        start_date (datetime.date): Start date
        end_date (datetime.date): End date
        chunk_days (int): Number of days per chunk (Open-Meteo allows ~2 months per request)
        
    Returns:
        pandas.DataFrame: Combined DataFrame with all data
    """
    all_data = []
    current_date = start_date
    
    while current_date < end_date:
        chunk_end = min(current_date + datetime.timedelta(days=chunk_days-1), end_date)
        
        # Format dates for API
        start_str = current_date.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        
        # Get data for this chunk
        chunk_data = get_weather_data(latitude, longitude, start_str, end_str)
        
        if chunk_data is not None:
            all_data.append(chunk_data)
            print(f"Successfully retrieved weather data from {start_str} to {end_str}")
        else:
            print(f"Failed to retrieve weather data from {start_str} to {end_str}")
        
        # Move to next chunk
        current_date = chunk_end + datetime.timedelta(days=1)
        
        # Add a small delay to avoid hitting API rate limits
        time.sleep(1)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def create_local_holiday_database():
    """
    Create a local database of Maharashtra festivals and holidays
    
    Returns:
        pandas.DataFrame: DataFrame with holiday information
    """
    # Major Maharashtra festivals and holidays for 2023-2025
    holidays = [
        # 2023
        {'date': '2023-01-14', 'name': 'Makar Sankranti', 'type': 'Hindu Festival'},
        {'date': '2023-01-26', 'name': 'Republic Day', 'type': 'National Holiday'},
        {'date': '2023-03-07', 'name': 'Holi', 'type': 'Hindu Festival'},
        {'date': '2023-03-22', 'name': 'Gudi Padwa', 'type': 'Maharashtra Festival'},
        {'date': '2023-04-14', 'name': 'Dr. Ambedkar Jayanti', 'type': 'National Holiday'},
        {'date': '2023-05-01', 'name': 'Maharashtra Day', 'type': 'State Holiday'},
        {'date': '2023-06-29', 'name': 'Ashadhi Ekadashi', 'type': 'Maharashtra Festival'},
        {'date': '2023-08-15', 'name': 'Independence Day', 'type': 'National Holiday'},
        {'date': '2023-09-19', 'name': 'Ganesh Chaturthi', 'type': 'Maharashtra Festival'},
        {'date': '2023-10-02', 'name': 'Gandhi Jayanti', 'type': 'National Holiday'},
        {'date': '2023-10-24', 'name': 'Diwali', 'type': 'Hindu Festival'},
        {'date': '2023-11-12', 'name': 'Kartiki Ekadashi', 'type': 'Maharashtra Festival'},
        
        # 2024
        {'date': '2024-01-15', 'name': 'Makar Sankranti', 'type': 'Hindu Festival'},
        {'date': '2024-01-26', 'name': 'Republic Day', 'type': 'National Holiday'},
        {'date': '2024-03-25', 'name': 'Holi', 'type': 'Hindu Festival'},
        {'date': '2024-04-09', 'name': 'Gudi Padwa', 'type': 'Maharashtra Festival'},
        {'date': '2024-04-14', 'name': 'Dr. Ambedkar Jayanti', 'type': 'National Holiday'},
        {'date': '2024-05-01', 'name': 'Maharashtra Day', 'type': 'State Holiday'},
        {'date': '2024-07-17', 'name': 'Ashadhi Ekadashi', 'type': 'Maharashtra Festival'},
        {'date': '2024-08-15', 'name': 'Independence Day', 'type': 'National Holiday'},
        {'date': '2024-09-07', 'name': 'Ganesh Chaturthi', 'type': 'Maharashtra Festival'},
        {'date': '2024-10-02', 'name': 'Gandhi Jayanti', 'type': 'National Holiday'},
        {'date': '2024-10-31', 'name': 'Diwali', 'type': 'Hindu Festival'},
        {'date': '2024-11-30', 'name': 'Kartiki Ekadashi', 'type': 'Maharashtra Festival'},
        
        # 2025
        {'date': '2025-01-14', 'name': 'Makar Sankranti', 'type': 'Hindu Festival'},
        {'date': '2025-01-26', 'name': 'Republic Day', 'type': 'National Holiday'},
        {'date': '2025-03-14', 'name': 'Holi', 'type': 'Hindu Festival'},
        {'date': '2025-03-29', 'name': 'Gudi Padwa', 'type': 'Maharashtra Festival'},
        {'date': '2025-04-14', 'name': 'Dr. Ambedkar Jayanti', 'type': 'National Holiday'},
        {'date': '2025-05-01', 'name': 'Maharashtra Day', 'type': 'State Holiday'},
        {'date': '2025-07-06', 'name': 'Ashadhi Ekadashi', 'type': 'Maharashtra Festival'},
        {'date': '2025-08-15', 'name': 'Independence Day', 'type': 'National Holiday'},
        {'date': '2025-08-28', 'name': 'Ganesh Chaturthi', 'type': 'Maharashtra Festival'},
        {'date': '2025-10-02', 'name': 'Gandhi Jayanti', 'type': 'National Holiday'},
        {'date': '2025-10-20', 'name': 'Diwali', 'type': 'Hindu Festival'},
        {'date': '2025-11-19', 'name': 'Kartiki Ekadashi', 'type': 'Maharashtra Festival'},
        
        # Add fasting days
        {'date': '2023-02-18', 'name': 'Maha Shivaratri', 'type': 'Fasting Day'},
        {'date': '2023-07-29', 'name': 'Guru Purnima', 'type': 'Fasting Day'},
        {'date': '2023-08-30', 'name': 'Janmashtami', 'type': 'Fasting Day'},
        {'date': '2023-09-28', 'name': 'Anant Chaturdashi', 'type': 'Fasting Day'},
        
        {'date': '2024-03-08', 'name': 'Maha Shivaratri', 'type': 'Fasting Day'},
        {'date': '2024-07-21', 'name': 'Guru Purnima', 'type': 'Fasting Day'},
        {'date': '2024-08-26', 'name': 'Janmashtami', 'type': 'Fasting Day'},
        {'date': '2024-09-16', 'name': 'Anant Chaturdashi', 'type': 'Fasting Day'},
        
        {'date': '2025-02-26', 'name': 'Maha Shivaratri', 'type': 'Fasting Day'},
        {'date': '2025-07-10', 'name': 'Guru Purnima', 'type': 'Fasting Day'},
        {'date': '2025-08-16', 'name': 'Janmashtami', 'type': 'Fasting Day'},
        {'date': '2025-09-06', 'name': 'Anant Chaturdashi', 'type': 'Fasting Day'},
    ]
    
    df = pd.DataFrame(holidays)
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Add primary_type column
    df['primary_type'] = df['type'].apply(lambda x: x.split()[0] if x else None)
    
    # Add has_festival column (1 for festival, 0 for no festival)
    df['has_festival'] = 1
    
    return df

def enrich_sales_data(sales_file, output_file):
    """
    Enrich sales data with weather and festival information
    
    Args:
        sales_file (str): Path to sales data CSV file
        output_file (str): Path to output CSV file
    """
    print(f"Reading sales data from {sales_file}...")
    sales_df = pd.read_csv(sales_file)
    
    # Convert date to datetime
    sales_df['date'] = pd.to_datetime(sales_df['date']).dt.date
    
    # Get unique date range from sales data
    min_date = sales_df['date'].min()
    max_date = sales_df['date'].max()
    
    # Format dates for API
    start_date_str = min_date.strftime('%Y-%m-%d')
    end_date_str = max_date.strftime('%Y-%m-%d')
    
    print(f"Sales data date range: {start_date_str} to {end_date_str}")
    
    # Get weather data
    latitude = 19.0948  # Mumbai Santacruz latitude
    longitude = 72.8471  # Mumbai Santacruz longitude
    
    weather_df = get_data_in_chunks(latitude, longitude, min_date, max_date)
    
    if weather_df is None:
        print("Failed to retrieve weather data. Exiting.")
        return
    
    # Get festival data
    festivals_df = create_local_holiday_database()
    print(f"Found {len(festivals_df)} festivals and holidays")
    
    # Create a date-hour key for merging
    sales_df['date_hour'] = [f"{d}_{h}" for d, h in zip(sales_df['date'], sales_df['hour'])]
    weather_df['date_hour'] = [f"{d}_{h}" for d, h in zip(weather_df['date'], weather_df['hour'])]
    
    # Merge sales with weather data
    print("Merging sales data with weather data...")
    merged_df = pd.merge(
        sales_df,
        weather_df[['date_hour', 'temp', 'humidity', 'precip', 'windspeed', 'pressure', 'cloudcover']],
        on='date_hour',
        how='left'
    )
    
    # Drop the temporary key
    merged_df = merged_df.drop(columns=['date_hour'])
    
    # Merge with festival data
    print("Merging with festival data...")
    enriched_df = pd.merge(
        merged_df,
        festivals_df[['date', 'name', 'type', 'primary_type', 'has_festival']],
        on='date',
        how='left'
    )
    
    # Fill missing festival data
    enriched_df['has_festival'] = enriched_df['has_festival'].fillna(0).astype(int)
    
    # Add day of week and month features
    enriched_df['day_of_week'] = pd.to_datetime(enriched_df['date']).dt.dayofweek
    enriched_df['month'] = pd.to_datetime(enriched_df['date']).dt.month
    enriched_df['is_weekend'] = enriched_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Save enriched data
    enriched_df.to_csv(output_file, index=False)
    print(f"Enriched sales data saved to {output_file}")
    
    # Print summary
    print("\nData Summary:")
    print(f"Total records: {len(enriched_df)}")
    print(f"Date range: {enriched_df['date'].min()} to {enriched_df['date'].max()}")
    print(f"Features included: {', '.join(enriched_df.columns)}")
    
    # Count records with festival data
    festival_records = enriched_df[enriched_df['has_festival'] == 1].shape[0]
    print(f"Records on festival days: {festival_records} ({festival_records/len(enriched_df)*100:.2f}%)")
    
    return enriched_df

def check_table_exists(client, project_id, dataset_id, table_id):
    """
    Check if a BigQuery table exists
    
    Args:
        client (bigquery.Client): BigQuery client
        project_id (str): Project ID
        dataset_id (str): Dataset ID
        table_id (str): Table ID
        
    Returns:
        bool: True if table exists, False otherwise
    """
    try:
        dataset_ref = client.dataset(dataset_id, project=project_id)
        table_ref = dataset_ref.table(table_id)
        client.get_table(table_ref)
        return True
    except Exception as e:
        print(f"Table check error: {str(e)}")
        return False

def get_sales_data_from_bigquery(credentials_path, start_date, end_date):
    """
    Fetch sales data from BigQuery
    
    Args:
        credentials_path (str): Path to the credentials JSON file
        start_date (str): Start date in format YYYY-MM-DD
        end_date (str): End date in format YYYY-MM-DD
        
    Returns:
        pandas.DataFrame: DataFrame containing sales data
    """
    # Load credentials
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    
    # Create BigQuery client
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    
    # SQL query to fetch only necessary fields
    project_id = credentials.project_id  # Use the project from credentials
    dataset = "Ettarra_Juhu"  # Update this if needed
    table = "SALES_MASTERDATA"  # Update this if needed
    
    # Print the project we're using
    print(f"Using project: {project_id}")
    
    # Check if table exists
    if not check_table_exists(client, project_id, dataset, table):
        print(f"Table `{project_id}.{dataset}.{table}` does not exist or you don't have access.")
        print("Available datasets and tables in your project:")
        
        # List available datasets
        datasets = list(client.list_datasets())
        if datasets:
            print("Datasets:")
            for dataset_item in datasets:
                dataset_id = dataset_item.dataset_id
                print(f"- {dataset_id}")
                
                # List tables in dataset
                tables = list(client.list_tables(dataset_item.reference))
                if tables:
                    print("  Tables:")
                    for table_item in tables:
                        print(f"  - {table_item.table_id}")
                else:
                    print("  No tables found in this dataset.")
        else:
            print("No datasets found in your project.")
            
        return None
    
    query = f"""
    SELECT 
        Date as date,
        Timestamp as timestamp,
        Item_Name as item_name,
        Price as price,
        Qty_ as quantity,
        Order_Type as order_type,
        Final_Total as final_total
    FROM 
        `{project_id}.{dataset}.{table}`
    WHERE 
        Date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY 
        Date, Timestamp
    """
    
    print(f"Fetching sales data from {start_date} to {end_date}...")
    query_job = client.query(query)
    
    # Convert to DataFrame
    df = query_job.to_dataframe()
    
    # Process the data
    if not df.empty:
        # Convert date to datetime.date for merging
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Extract hour from timestamp for hourly analysis
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        # Categorize order types
        df['is_dine_in'] = df['order_type'].str.lower().str.contains('dine').fillna(False)
        df['is_online'] = ~df['is_dine_in']
    
    return df

def main():
    # Input and output files
    sales_file = "sales_data_2023_2025.csv"
    output_file = "enriched_sales_data_2023_2025.csv"
    credentials_path = "creds.json"
    
    # Choose data source
    use_bigquery = input("Fetch fresh data from BigQuery? (y/n): ").lower() == 'y'
    
    if use_bigquery:
        # Get date range for BigQuery query
        today = datetime.datetime.now().date()
        days_back = int(input("How many days of historical data to fetch? (default: 30): ") or "30")
        days_forward = int(input("How many days of future data to include? (default: 0): ") or "0")
        
        start_date = (today - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = (today + datetime.timedelta(days=days_forward)).strftime('%Y-%m-%d')
        
        # Check if credentials file exists
        if not os.path.exists(credentials_path):
            print(f"Credentials file {credentials_path} not found.")
            return
            
        # Get sales data from BigQuery
        try:
            sales_df = get_sales_data_from_bigquery(credentials_path, start_date, end_date)
            if sales_df is None or len(sales_df) == 0:
                print("No sales data retrieved from BigQuery. Exiting.")
                return
                
            print(f"Retrieved {len(sales_df)} sales records from BigQuery.")
            
            # Save to CSV
            sales_df.to_csv(sales_file, index=False)
            print(f"Sales data saved to {sales_file}")
            
            # Enrich the data
            enrich_sales_data(sales_file, output_file)
        except Exception as e:
            print(f"Error fetching data from BigQuery: {str(e)}")
            print("Falling back to local file...")
            use_bigquery = False
    
    if not use_bigquery:
        # Check if sales file exists
        if not os.path.exists(sales_file):
            print(f"Sales file {sales_file} not found.")
            return
        
        # Enrich sales data from local file
        enrich_sales_data(sales_file, output_file)
    
    print("\nProcess complete!")

if __name__ == "__main__":
    main()