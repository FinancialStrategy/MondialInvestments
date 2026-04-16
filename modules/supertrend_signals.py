"""
Supertrend Indicator with Buy/Sell Signal Generation
Professional Trading Strategy Module - No TA-Lib Required
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit as st
from typing import Optional, Dict, List

class SupertrendAnalyzer:
    """
    Supertrend Indicator Implementation with Signal Generation
    
    Parameters:
    - period: ATR period (default: 10)
    - multiplier: ATR multiplier (default: 3.0)
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self.df = None
        self.signals = None
        
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR without TA-Lib"""
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        
        return atr
        
    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend indicator
        """
        df_copy = df.copy()
        
        # Calculate ATR
        df_copy['ATR'] = self.calculate_atr(df_copy)
        
        # Calculate Basic Bands
        hl_avg = (df_copy['High'] + df_copy['Low']) / 2
        df_copy['Basic_Upper'] = hl_avg + (self.multiplier * df_copy['ATR'])
        df_copy['Basic_Lower'] = hl_avg - (self.multiplier * df_copy['ATR'])
        
        # Calculate Final Bands
        df_copy['Final_Upper'] = 0.0
        df_copy['Final_Lower'] = 0.0
        df_copy['Supertrend'] = 0.0
        df_copy['Trend'] = 1  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(df_copy)):
            # Upper Band
            if (df_copy['Basic_Upper'].iloc[i] < df_copy['Final_Upper'].iloc[i-1]) or \
               (df_copy['Close'].iloc[i-1] > df_copy['Final_Upper'].iloc[i-1]):
                df_copy.loc[df_copy.index[i], 'Final_Upper'] = df_copy['Basic_Upper'].iloc[i]
            else:
                df_copy.loc[df_copy.index[i], 'Final_Upper'] = df_copy['Final_Upper'].iloc[i-1]
            
            # Lower Band
            if (df_copy['Basic_Lower'].iloc[i] > df_copy['Final_Lower'].iloc[i-1]) or \
               (df_copy['Close'].iloc[i-1] < df_copy['Final_Lower'].iloc[i-1]):
                df_copy.loc[df_copy.index[i], 'Final_Lower'] = df_copy['Basic_Lower'].iloc[i]
            else:
                df_copy.loc[df_copy.index[i], 'Final_Lower'] = df_copy['Final_Lower'].iloc[i-1]
            
            # Determine Supertrend and Trend
            if df_copy['Close'].iloc[i] <= df_copy['Final_Upper'].iloc[i]:
                df_copy.loc[df_copy.index[i], 'Supertrend'] = df_copy['Final_Upper'].iloc[i]
                df_copy.loc[df_copy.index[i], 'Trend'] = -1
            else:
                df_copy.loc[df_copy.index[i], 'Supertrend'] = df_copy['Final_Lower'].iloc[i]
                df_copy.loc[df_copy.index[i], 'Trend'] = 1
        
        self.df = df_copy
        return df_copy
    
    def generate_signals(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Generate Buy and Sell signals based on Supertrend"""
        if df is not None:
            self.calculate_supertrend(df)
        elif self.df is None:
            raise ValueError("No data provided. Please provide DataFrame or run calculate_supertrend first.")
        
        signals_df = self.df.copy()
        
        # Generate signals
        signals_df['Signal'] = 0
        signals_df['Buy_Signal'] = np.nan
        signals_df['Sell_Signal'] = np.nan
        
        # Detect trend changes
        signals_df['Trend_Change'] = signals_df['Trend'].diff()
        
        # Buy signal: Trend changes from -1 to 1
        buy_condition = (signals_df['Trend_Change'] == 2) & (signals_df['Trend'] == 1)
        signals_df.loc[buy_condition, 'Signal'] = 1
        signals_df.loc[buy_condition, 'Buy_Signal'] = signals_df['Close']
        
        # Sell signal: Trend changes from 1 to -1
        sell_condition = (signals_df['Trend_Change'] == -2) & (signals_df['Trend'] == -1)
        signals_df.loc[sell_condition, 'Signal'] = -1
        signals_df.loc[sell_condition, 'Sell_Signal'] = signals_df['Close']
        
        # Calculate position (for backtesting)
        signals_df['Position'] = signals_df['Signal'].replace(to_replace=0, method='ffill').fillna(0)
        
        # Calculate returns
        signals_df['Strategy_Returns'] = signals_df['Close'].pct_change() * signals_df['Position'].shift(1)
        signals_df['Buy_Hold_Returns'] = signals_df['Close'].pct_change()
        
        # Calculate cumulative returns
        signals_df['Cumulative_Strategy'] = (1 + signals_df['Strategy_Returns']).cumprod()
        signals_df['Cumulative_BuyHold'] = (1 + signals_df['Buy_Hold_Returns']).cumprod()
        
        self.signals = signals_df
        return signals_df
    
    def calculate_performance_metrics(self) -> dict:
        """Calculate performance metrics for the strategy"""
        if self.signals is None:
            raise ValueError("Please generate signals first using generate_signals()")
        
        total_trades = len(self.signals[self.signals['Signal'] != 0])
        buy_trades = len(self.signals[self.signals['Signal'] == 1])
        sell_trades = len(self.signals[self.signals['Signal'] == -1])
        
        # Calculate trade returns
        trade_returns = []
        entry_price = 0
        
        for i in range(len(self.signals)):
            if self.signals['Signal'].iloc[i] == 1:
                entry_price = self.signals['Close'].iloc[i]
            elif self.signals['Signal'].iloc[i] == -1 and entry_price > 0:
                exit_price = self.signals['Close'].iloc[i]
                trade_return = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_return)
                entry_price = 0
        
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
        
        # Strategy vs Buy & Hold
        total_strategy_return = self.signals['Cumulative_Strategy'].iloc[-1] - 1
        total_buyhold_return = self.signals['Cumulative_BuyHold'].iloc[-1] - 1
        
        # Sharpe Ratio
        strategy_returns = self.signals['Strategy_Returns'].dropna()
        sharpe = (strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() != 0 else 0
        
        # Max Drawdown
        cumulative = self.signals['Cumulative_Strategy']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'Total Trades': total_trades,
            'Buy Signals': buy_trades,
            'Sell Signals': sell_trades,
            'Average Trade Return': avg_trade_return,
            'Win Rate': win_rate,
            'Strategy Total Return': total_strategy_return,
            'Buy & Hold Return': total_buyhold_return,
            'Strategy Alpha': total_strategy_return - total_buyhold_return,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Total Days': len(self.signals)
        }
        
        return metrics
    
    def create_signal_chart(self, title: str = "Supertrend Strategy") -> go.Figure:
        """Create interactive chart with Supertrend and signals"""
        if self.signals is None:
            raise ValueError("Please generate signals first")
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(title, "Supertrend & Trend", "Cumulative Returns Comparison")
        )
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals['Close'],
                name='Close Price',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
        
        # Supertrend line
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals['Supertrend'],
                name='Supertrend',
                line=dict(color='purple', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Buy signals
        buy_signals = self.signals[self.signals['Signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Close'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(symbol='triangle-up', size=14, color='green'),
                    text=[f'BUY at {price:.2f}' for price in buy_signals['Close']]
                ),
                row=1, col=1
            )
        
        # Sell signals
        sell_signals = self.signals[self.signals['Signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['Close'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(symbol='triangle-down', size=14, color='red'),
                    text=[f'SELL at {price:.2f}' for price in sell_signals['Close']]
                ),
                row=1, col=1
            )
        
        # Trend background
        for i in range(len(self.signals) - 1):
            if self.signals['Trend'].iloc[i] == 1:
                fillcolor = 'rgba(46, 204, 113, 0.15)'
            else:
                fillcolor = 'rgba(231, 76, 60, 0.15)'
            
            fig.add_vrect(
                x0=self.signals.index[i],
                x1=self.signals.index[i+1],
                fillcolor=fillcolor,
                opacity=0.3,
                line_width=0,
                row=2, col=1
            )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals['Cumulative_Strategy'],
                name='Strategy Returns',
                line=dict(color='blue', width=2.5)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals['Cumulative_BuyHold'],
                name='Buy & Hold',
                line=dict(color='gray', width=2, dash='dash')
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=900,
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Returns", tickformat=".0%", row=3, col=1)
        
        return fig
    
    def create_performance_dashboard(self) -> None:
        """Create Streamlit dashboard for performance metrics"""
        metrics = self.calculate_performance_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", metrics['Total Trades'])
            st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
        
        with col2:
            st.metric("Strategy Return", f"{metrics['Strategy Total Return']:.2%}")
            st.metric("Buy & Hold", f"{metrics['Buy & Hold Return']:.2%}")
        
        with col3:
            st.metric("Strategy Alpha", f"{metrics['Strategy Alpha']:.2%}")
            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
            st.metric("Avg Trade Return", f"{metrics['Average Trade Return']:.2%}")
        
        # Trade log
        trades = []
        entry_price = None
        entry_date = None
        
        for i in range(len(self.signals)):
            if self.signals['Signal'].iloc[i] == 1:
                entry_price = self.signals['Close'].iloc[i]
                entry_date = self.signals.index[i]
            elif self.signals['Signal'].iloc[i] == -1 and entry_price is not None:
                exit_price = self.signals['Close'].iloc[i]
                exit_date = self.signals.index[i]
                trade_return = (exit_price - entry_price) / entry_price
                holding_days = (exit_date - entry_date).days
                
                trades.append({
                    'Entry Date': entry_date.strftime('%Y-%m-%d'),
                    'Exit Date': exit_date.strftime('%Y-%m-%d'),
                    'Entry Price': f"{entry_price:.2f}",
                    'Exit Price': f"{exit_price:.2f}",
                    'Return': f"{trade_return:.2%}",
                    'Holding Days': holding_days
                })
                entry_price = None
        
        if trades:
            st.subheader("📋 Trade Log")
            st.dataframe(pd.DataFrame(trades), use_container_width=True)


def scan_multiple_stocks(tickers: list, df_dict: dict, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Scan multiple stocks for Supertrend signals"""
    results = []
    
    for ticker in tickers:
        if ticker in df_dict:
            df = df_dict[ticker]
            if len(df) > period:
                try:
                    analyzer = SupertrendAnalyzer(period, multiplier)
                    signals = analyzer.generate_signals(df)
                    
                    latest_trend = signals['Trend'].iloc[-1]
                    latest_price = signals['Close'].iloc[-1]
                    latest_supertrend = signals['Supertrend'].iloc[-1]
                    
                    if latest_trend == 1:
                        action = "🟢 BUY" if signals['Signal'].iloc[-1] == 1 else "✅ HOLD"
                    else:
                        action = "🔴 SELL" if signals['Signal'].iloc[-1] == -1 else "⚠️ AVOID"
                    
                    results.append({
                        'Ticker': ticker,
                        'Action': action,
                        'Price': f"{latest_price:.2f}",
                        'Supertrend': f"{latest_supertrend:.2f}",
                        'Trend': 'UPTREND' if latest_trend == 1 else 'DOWNTREND'
                    })
                except Exception as e:
                    print(f"Error: {e}")
    
    return pd.DataFrame(results)
