import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import config  # Configuration module for database settings

def query_yield_curve(db_file: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Query yield curve data from SQLite database.

    Args:
        db_file (str): Path to SQLite database.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with columns ['date', 'maturity', 'yield'].
    """
    query = """
        SELECT * FROM BondYields
        WHERE date BETWEEN ? AND ?
    """
    try:
        with sqlite3.connect(db_file) as conn:
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        df = df.dropna()  # Remove missing values
        df = df.melt(id_vars=["date"], var_name="maturity", value_name="yield")  # Convert to long format
        df["date"] = pd.to_datetime(df["date"])  # Convert date column to datetime
        
        return df
    except Exception as e:
        print(f"Error querying yield curve data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def query_vix_data(db_file: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Query VIX data from SQLite database.

    Args:
        db_file (str): Path to SQLite database.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with VIX data, or an empty DataFrame on failure.
    """
    query = """
        SELECT * FROM VIX
        WHERE date BETWEEN ? AND ?
    """
    try:
        with sqlite3.connect(db_file) as conn:
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])  # Convert date column to datetime
        
        return df
    except Exception as e:
        print(f"Error querying VIX data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def get_historical_inflation(db_file: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical inflation data and compute Year-over-Year (YoY) CPI change.

    Args:
        db_file (str): Path to SQLite database.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with ['date', 'CPI', 'inflation_rate'].
    """
    query = """
        SELECT date, CPI
        FROM Inflation
        WHERE date BETWEEN ? AND ?
        ORDER BY date ASC
    """
    try:
        with sqlite3.connect(db_file) as conn:
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        df = df.sort_values("date")
        df["inflation_rate"] = df["CPI"].pct_change(12) * 100  # Compute YoY inflation rate
        
        return df.dropna()
    except Exception as e:
        print(f"Error querying inflation data: {e}")
        return pd.DataFrame()

def convert_maturity(maturity: str) -> float:
    """
    Convert maturity strings like 'T1', 'T1M', 'T1Y', '1M', '10Y' into numeric years.

    Args:
        maturity (str): Maturity string.

    Returns:
        float: Maturity in years.
    """
    maturity = maturity.strip().upper()
    
    if maturity.startswith("T"):
        maturity = maturity[1:]  # Remove leading 'T'

    if maturity.endswith("Y") and maturity[:-1].isdigit():
        return float(maturity[:-1])
    elif maturity.endswith("M") and maturity[:-1].isdigit():
        return float(maturity[:-1]) / 12
    elif maturity.isdigit():  # Handle cases like 'T1' (assuming months)
        return float(maturity) / 12

    raise ValueError(f"Unknown maturity format: {maturity}")

def prepare_data(db_file: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Prepare yield curve data for analysis.

    Args:
        db_file (str): Path to SQLite database.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Pivot table with dates as index and maturities as columns.
    """
    df = query_yield_curve(db_file, start_date, end_date)
    
    if df.empty:
        return df  # Return empty DataFrame if no data
    
    df["maturity_numeric"] = df["maturity"].apply(convert_maturity).round(4)
    df_pivot = df.pivot(index="date", columns="maturity_numeric", values="yield")
    
    return df_pivot.dropna(how="all", axis=1)  # Remove empty columns

def update_database(db_file: str, table_name: str, new_data: pd.DataFrame):
    """
    Generalized function to update any database table with new data.

    Args:
        db_file (str): Path to SQLite database.
        table_name (str): Name of the table to update.
        new_data (pd.DataFrame): DataFrame with new records.

    Returns:
        None
    """
    if new_data.empty:
        print(f"No new data to update in {table_name}.")
        return

    try:
        with sqlite3.connect(db_file) as conn:
            new_data.to_sql(table_name, conn, if_exists="append", index=False)
        
        print(f"Updated {table_name} database with {len(new_data)} new records.")
    except Exception as e:
        print(f"Error updating {table_name} database: {e}")

# Wrapper functions using update_database.py
def update_yields_database(db_file: str, new_data: pd.DataFrame):
    """ Update the BondYields table with new yield data. """
    update_database(db_file, "BondYields", new_data)

def update_vix_database(db_file: str, new_data: pd.DataFrame):
    """ Update the VIX table with new VIX data. """
    update_database(db_file, "VIX", new_data)

def update_inflation_database(db_file: str, new_data: pd.DataFrame):
    """ Update the Inflation table with new inflation data. """
    update_database(db_file, "Inflation", new_data)
