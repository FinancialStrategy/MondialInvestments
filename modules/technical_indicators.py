"""
Technical Indicators Module with TA-Lib Integration
"""

import pandas as pd
import numpy as np
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # ✅ FIXED: Import eklendi
from typing import Dict, Optional

def add_technical_indicators(df: pd.DataFrame, indicator_config: Dict) -> pd.DataFrame:
    """
    TA-Lib kullanarak teknik indikatörleri hesaplar
    
    Supported Indicators:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - ATR (Average True Range)
    """
    df_copy = df.copy()
    
    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in df_copy.columns]
    
    if missing_cols:
        # Try to create missing columns from Close price
        for col in missing_cols:
            if col == 'Volume':
                df_copy[col] = 0
            elif 'Close' in df_copy.columns:
                df_copy[col] = df_copy['Close']
            else:
                raise ValueError(f"Missing required column: {col}")
    
    # Convert to numeric and handle NaN
    for col in ['Open', 'High', 'Low', 'Close']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    # SMA (Simple Moving Average)
    if indicator_config.get('sma', False):
        df_copy['SMA_20'] = talib.SMA(df_copy['Close'], timeperiod=20)
        df_copy['SMA_50'] = talib.SMA(df_copy['Close'], timeperiod=50)
        df_copy['SMA_200'] = talib.SMA(df_copy['Close'], timeperiod=200)
    
    # EMA (Exponential Moving Average)
    if indicator_config.get('ema', False):
        df_copy['EMA_12'] = talib.EMA(df_copy['Close'], timeperiod=12)
        df_copy['EMA_26'] = talib.EMA(df_copy['Close'], timeperiod=26)
    
    # RSI (Relative Strength Index)
    if indicator_config.get('rsi', False):
        df_copy['RSI_14'] = talib.RSI(df_copy['Close'], timeperiod=14)
    
    # MACD (Moving Average Convergence Divergence)
    if indicator_config.get('macd', False):
        macd, signal, hist = talib.MACD(
            df_copy['Close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        df_copy['MACD'] = macd
        df_copy['MACD_Signal'] = signal
        df_copy['MACD_Hist'] = hist
    
    # Bollinger Bands
    if indicator_config.get('bollinger', False):
        upper, middle, lower = talib.BBANDS(
            df_copy['Close'], 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        df_copy['BB_Upper'] = upper
        df_copy['BB_Middle'] = middle
        df_copy['BB_Lower'] = lower
    
    # ATR (Average True Range)
    if indicator_config.get('atr', False):
        df_copy['ATR_14'] = talib.ATR(
            df_copy['High'], 
            df_copy['Low'], 
            df_copy['Close'], 
            timeperiod=14
        )
    
    return df_copy

def create_candlestick_with_indicators(df: pd.DataFrame, ticker: str, indicators: Dict):
    """Interactive OHLC chart with technical indicators"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{ticker} - Price & Indicators', 'RSI', 'MACD')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if 'SMA_20' in df.columns and not df['SMA_20'].isna().all():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                      line=dict(color='orange', width=1.5)),
            row=1, col=1
        )
    if 'SMA_50' in df.columns and not df['SMA_50'].isna().all():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                      line=dict(color='blue', width=1.5)),
            row=1, col=1
        )
    if 'SMA_200' in df.columns and not df['SMA_200'].isna().all():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', 
                      line=dict(color='red', width=1.5)),
            row=1, col=1
        )
    if 'EMA_12' in df.columns and not df['EMA_12'].isna().all():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA_12'], name='EMA 12', 
                      line=dict(color='green', width=1.5, dash='dash')),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'BB_Upper' in df.columns and not df['BB_Upper'].isna().all():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                      line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                      line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
    
    # RSI
    if 'RSI_14' in df.columns and not df['RSI_14'].isna().all():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI_14'], name='RSI 14', 
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     row=2, col=1, annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     row=2, col=1, annotation_text="Oversold")
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns and not df['MACD'].isna().all():
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                      line=dict(color='blue', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', 
                      line=dict(color='red', width=2)),
            row=3, col=1
        )
        
        # Histogram
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', 
                  marker_color=colors),
            row=3, col=1
        )
        fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    fig.update_layout(
        title=f'{ticker} - Interactive Technical Analysis',
        xaxis_title='Date',
        yaxis_title='Price (USD/TRY/EUR)',
        height=800,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig
