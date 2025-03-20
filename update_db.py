import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fredapi import Fred
from dotenv import load_dotenv
import data
import config
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='update_log.txt'
)
logger = logging.getLogger('update_db')

# Load environment variables from .env file (if exists)
load_dotenv()

# Get FRED API key from environment variable
FRED_API_KEY = os.environ.get('FRED_API_KEY')
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY environment variable not set. Please set it before running this script.")

# Create FRED API client
fred = Fred(api_key=FRED_API_KEY)

def fetch_treasury_yields(start_date, end_date):
    """
    Fetch Treasury yield data from FRED for various maturities.
    
    Returns:
        DataFrame with date index and columns for different maturities.
    """
    logger.info(f"Fetching Treasury yields from {start_date} to {end_date}")
    
    # FRED series IDs for different Treasury maturities
    # Format: (maturity_label, series_id)
    treasury_series = [
        ('T1M', 'DGS1MO'),   # 1-Month Treasury Constant Maturity Rate
        ('T3M', 'DGS3MO'),   # 3-Month Treasury Constant Maturity Rate
        ('T6M', 'DGS6MO'),   # 6-Month Treasury Constant Maturity Rate
        ('T1', 'DGS1'),      # 1-Year Treasury Constant Maturity Rate
        ('T2', 'DGS2'),      # 2-Year Treasury Constant Maturity Rate
        ('T5', 'DGS5'),      # 5-Year Treasury Constant Maturity Rate
        ('T10', 'DGS10'),    # 10-Year Treasury Constant Maturity Rate
        ('T30', 'DGS30')     # 30-Year Treasury Constant Maturity Rate
    ]
    
    # Initialize dictionary to store series data
    yields_data = {}
    
    # Fetch each series
    for label, series_id in treasury_series:
        try:
            # Get data from FRED API
            series = fred.get_series(series_id, start_date, end_date)
            if not series.empty:
                yields_data[label] = series
                logger.info(f"Successfully fetched {label} yields ({len(series)} observations)")
            else:
                logger.warning(f"No data returned for {label} (series ID: {series_id})")
        except Exception as e:
            logger.error(f"Error fetching {label} (series ID: {series_id}): {str(e)}")
    
    # Convert to DataFrame
    if yields_data:
        df = pd.DataFrame(yields_data)
        df.index.name = 'date'
        
        # Convert index to datetime if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        return df
    else:
        logger.error("Failed to fetch any Treasury yield data")
        return pd.DataFrame()

def fetch_vix_data(start_date, end_date):
    """
    Fetch VIX index data from FRED.
    
    Returns:
        DataFrame with date, vix columns.
    """
    logger.info(f"Fetching VIX data from {start_date} to {end_date}")
    
    try:
        # VIX series ID in FRED is 'VIXCLS'
        vix_series = fred.get_series('VIXCLS', start_date, end_date)
        
        if vix_series.empty:
            logger.warning("No VIX data returned from FRED API")
            return pd.DataFrame()
        
        # Convert Series to DataFrame
        vix_df = vix_series.to_frame('vix')
        vix_df = vix_df.reset_index()
        vix_df.columns = ['date', 'vix']
        
        # Convert date to datetime if not already
        vix_df['date'] = pd.to_datetime(vix_df['date'])
        
        # Fill any missing values using forward fill method
        vix_df = vix_df.fillna(method='ffill')
        
        logger.info(f"Successfully fetched VIX data ({len(vix_df)} observations)")
        return vix_df
    
    except Exception as e:
        logger.error(f"Error fetching VIX data: {str(e)}")
        return pd.DataFrame()

def fetch_inflation_data(start_date, end_date):
    """
    Fetch CPI (Consumer Price Index) data from FRED for inflation calculations.
    
    Returns:
        DataFrame with date, CPI columns.
    """
    logger.info(f"Fetching CPI data from {start_date} to {end_date}")
    
    try:
        # CPI series ID in FRED is 'CPIAUCSL' (Consumer Price Index for All Urban Consumers)
        cpi_series = fred.get_series('CPIAUCSL', start_date, end_date)
        
        if cpi_series.empty:
            logger.warning("No CPI data returned from FRED API")
            return pd.DataFrame()
        
        # Convert Series to DataFrame
        cpi_df = cpi_series.to_frame('CPI')
        cpi_df = cpi_df.reset_index()
        cpi_df.columns = ['date', 'CPI']
        
        # Convert date to datetime if not already
        cpi_df['date'] = pd.to_datetime(cpi_df['date'])
        
        logger.info(f"Successfully fetched CPI data ({len(cpi_df)} observations)")
        return cpi_df
    
    except Exception as e:
        logger.error(f"Error fetching CPI data: {str(e)}")
        return pd.DataFrame()

def prepare_yields_for_db(yields_df):
    """
    Prepare Treasury yields data for the database format.
    
    Returns:
        DataFrame formatted for the BondYields table.
    """
    if yields_df.empty:
        return pd.DataFrame()
    
    # Reset index to get date as a column
    df = yields_df.reset_index()
    
    # Convert date to string format for SQLite compatibility
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    return df

def update_database():
    """
    Main function to update all database tables with latest data from FRED.
    """
    logger.info("Starting database update process")
    
    # Get default date parameters
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Try to get last date in database to use as start_date
    try:
        with sqlite3.connect(config.DB_FILE) as conn:
            # Check BondYields table
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(date) FROM BondYields")
            last_date = cursor.fetchone()[0]
            
            if last_date:
                # Start from the day after the last date in database
                start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                # If table is empty or doesn't exist, use default start date
                start_date = config.DEFAULT_START_DATE
    except Exception as e:
        logger.warning(f"Error determining start date: {str(e)}. Using default start date.")
        start_date = config.DEFAULT_START_DATE
    
    logger.info(f"Update period: {start_date} to {end_date}")
    
    # Fetch data from FRED
    yields_df = fetch_treasury_yields(start_date, end_date)
    vix_df = fetch_vix_data(start_date, end_date)
    cpi_df = fetch_inflation_data(start_date, end_date)
    
    # Prepare and update database tables
    if not yields_df.empty:
        yields_db_df = prepare_yields_for_db(yields_df)
        data.update_yields_database(config.DB_FILE, yields_db_df)
    else:
        logger.warning("No Treasury yields data to update")
    
    if not vix_df.empty:
        data.update_vix_database(config.DB_FILE, vix_df)
    else:
        logger.warning("No VIX data to update")
    
    if not cpi_df.empty:
        data.update_inflation_database(config.DB_FILE, cpi_df)
    else:
        logger.warning("No CPI data to update")
    
    logger.info("Database update completed")

def initialize_database():
    """
    Initialize database with historical data for the first time.
    """
    logger.info("Initializing database with historical data")
    
    # Use wide historical range for initial setup
    start_date = '2000-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch historical data
    yields_df = fetch_treasury_yields(start_date, end_date)
    vix_df = fetch_vix_data(start_date, end_date)
    cpi_df = fetch_inflation_data(start_date, end_date)
    
    # Prepare and update database tables
    if not yields_df.empty:
        yields_db_df = prepare_yields_for_db(yields_df)
        data.update_yields_database(config.DB_FILE, yields_db_df)
        logger.info(f"Initialized BondYields table with {len(yields_db_df)} records")
    else:
        logger.error("Failed to initialize BondYields table - no data")
    
    if not vix_df.empty:
        data.update_vix_database(config.DB_FILE, vix_df)
        logger.info(f"Initialized VIX table with {len(vix_df)} records")
    else:
        logger.error("Failed to initialize VIX table - no data")
    
    if not cpi_df.empty:
        data.update_inflation_database(config.DB_FILE, cpi_df)
        logger.info(f"Initialized Inflation table with {len(cpi_df)} records")
    else:
        logger.error("Failed to initialize Inflation table - no data")
    
    logger.info("Database initialization completed")

if __name__ == "__main__":
    import sqlite3
    import argparse
    
    parser = argparse.ArgumentParser(description='Update Treasury Yields database with FRED data')
    parser.add_argument('--init', action='store_true', help='Initialize database with historical data')
    parser.add_argument('--force-update', action='store_true', help='Force update with latest data regardless of last update date')
    
    args = parser.parse_args()
    
    if args.init:
        # Create database tables if they don't exist
        with sqlite3.connect(config.DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Create BondYields table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS BondYields (
                date TEXT PRIMARY KEY,
                T1M REAL,
                T3M REAL,
                T6M REAL,
                T1 REAL,
                T2 REAL,
                T5 REAL,
                T10 REAL,
                T30 REAL
            )
            ''')
            
            # Create VIX table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS VIX (
                date TEXT PRIMARY KEY,
                vix REAL,
                vix_ma5 REAL,
                vix_ma20 REAL,
                vix_ma60 REAL,
                vix_percentile REAL,
                vix_change REAL,
                vix_volatility REAL,
                high_vol_regime INTEGER,
                medium_vol_regime INTEGER,
                low_vol_regime INTEGER,
                last_updated TEXT
            )
            ''')
            
            # Create Inflation table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Inflation (
                date TEXT PRIMARY KEY,
                CPI REAL
            )
            ''')
            
            conn.commit()
        
        # Initialize with historical data
        initialize_database()
    else:
        # Regular update
        update_database()