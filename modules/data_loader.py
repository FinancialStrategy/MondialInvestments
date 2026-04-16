"""
Global Data Loader Module - With Timezone Support
Fixed for Streamlit Cloud deployment
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List
import yaml
from datetime import datetime
import pytz
import time

@st.cache_data(ttl=3600, show_spinner=False)
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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data with timezone support for Streamlit Cloud
    """
    if not tickers:
        return {}
    
    ticker_data = {}
    
    # Set timezone to UTC to avoid "No timezone found" error
    tz = pytz.UTC
    
    # Convert dates to timezone-aware datetime
    if isinstance(start_date, str):
        start = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start = start_date
    
    if isinstance(end_date, str):
        end = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end = end_date
    
    start_aware = tz.localize(start)
    end_aware = tz.localize(end)
    
    for ticker in tickers:
        try:
            import yfinance as yf
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            
            # Download with timezone-aware dates
            data = yf.download(
                ticker,
                start=start_aware,
                end=end_aware,
                progress=False,
                auto_adjust=True,
                ignore_tz=False  # Important: Don't ignore timezone
            )
            
            if not data.empty and 'Close' in data.columns:
                # Ensure index is timezone-naive for Streamlit
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                ticker_data[ticker] = data
            else:
                st.warning(f"No data returned for {ticker}")
                
        except Exception as e:
            st.warning(f"Could not load {ticker}: {e}")
            continue
    
    return ticker_data

def get_benchmark_data(benchmark: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch benchmark index data with timezone support"""
    try:
        import yfinance as yf
        import pytz
        
        tz = pytz.UTC
        
        if isinstance(start_date, str):
            start = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start = start_date
            
        if isinstance(end_date, str):
            end = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end = end_date
        
        start_aware = tz.localize(start)
        end_aware = tz.localize(end)
        
        bench = yf.download(
            benchmark, 
            start=start_aware, 
            end=end_aware, 
            progress=False,
            ignore_tz=False
        )
        
        if not bench.empty and 'Close' in bench.columns:
            if bench.index.tz is not None:
                bench.index = bench.index.tz_localize(None)
            return bench['Close']
            
    except Exception as e:
        st.warning(f"Could not fetch benchmark {benchmark}: {e}")
    
    return pd.Series()
