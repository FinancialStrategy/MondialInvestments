"""
Global Data Loader Module
Supports Turkish, US, European, Asian, and Australian markets
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional
import yaml
from datetime import datetime

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_config():
    """Load global configuration from config.yaml"""
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("❌ config.yaml file not found!")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading config: {e}")
        return {}

@st.cache_data(ttl=3600)
def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple tickers using yfinance.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Dictionary with ticker as key and OHLC DataFrame as value
    """
    if not tickers:
        return {}
    
    try:
        # Download data with progress indicator
        with st.spinner(f"📥 Fetching data for {len(tickers)} stocks..."):
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=True,
                progress=False,
                threads=True
            )
        
        ticker_data = {}
        
        # Handle single ticker case
        if len(tickers) == 1:
            ticker = tickers[0]
            if not data.empty:
                # Ensure it has OHLC columns
                if isinstance(data, pd.DataFrame):
                    if 'Close' in data.columns:
                        # Single ticker returns DataFrame with OHLC columns
                        ticker_data[ticker] = data
                    else:
                        # Create a proper DataFrame
                        ticker_data[ticker] = pd.DataFrame({
                            'Open': data['Open'] if 'Open' in data else data['Close'],
                            'High': data['High'] if 'High' in data else data['Close'],
                            'Low': data['Low'] if 'Low' in data else data['Close'],
                            'Close': data['Close'] if 'Close' in data else data,
                            'Volume': data['Volume'] if 'Volume' in data else 0
                        }, index=data.index)
        else:
            # Multiple tickers case
            for ticker in tickers:
                if ticker in data.columns.levels[0]:
                    df = data[ticker].copy()
                    # Ensure all required columns exist
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col not in df.columns:
                            if col == 'Volume':
                                df[col] = 0
                            elif 'Close' in df.columns:
                                df[col] = df['Close']
                            else:
                                df[col] = 0
                    ticker_data[ticker] = df
        
        # Clean and validate data
        valid_data = {}
        for ticker, df in ticker_data.items():
            if df is not None and not df.empty and len(df) > 10:
                # Drop NaN values
                df_clean = df.dropna()
                if len(df_clean) > 0:
                    valid_data[ticker] = df_clean
        
        if not valid_data:
            st.warning(f"⚠️ No valid data found for tickers: {tickers}")
        
        return valid_data
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def get_benchmark_data(benchmark: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch benchmark index data for comparison"""
    try:
        # Fix common benchmark symbols
        benchmark_map = {
            "XU100.IS": "^XU100",
            "XU100": "^XU100",
            "GSPC": "^GSPC",
            "IXIC": "^IXIC",
            "DJI": "^DJI"
        }
        
        symbol = benchmark_map.get(benchmark, benchmark)
        
        with st.spinner(f"📊 Fetching benchmark data for {benchmark}..."):
            bench = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if not bench.empty and 'Close' in bench.columns:
            return bench['Close']
        else:
            st.warning(f"⚠️ No benchmark data for {benchmark}")
            return pd.Series()
            
    except Exception as e:
        st.warning(f"⚠️ Could not fetch benchmark {benchmark}: {e}")
        return pd.Series()

def get_all_markets_data(selected_indices: List[str], start_date: str, end_date: str) -> Dict:
    """Fetch data from multiple global markets"""
    config = load_config()
    all_data = {}
    
    if not config:
        return all_data
    
    for index_name in selected_indices:
        if index_name not in config.get('indices', {}):
            continue
            
        index_config = config['indices'][index_name]
        tickers = index_config.get('tickers', [])
        
        if not tickers:
            continue
        
        st.info(f"🌍 Loading {index_name} market data...")
        data = fetch_market_data(tickers, start_date, end_date)
        
        if data:
            all_data[index_name] = {
                'data': data,
                'config': index_config
            }
    
    return all_data
