"""
Global Equity Analytics Platform
Professional Multi-Market Portfolio Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz

# Page configuration
st.set_page_config(
    page_title="Global Equity Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple custom CSS - minimal
st.markdown("""
<style>
.stMetric {
    background-color: #f8f9fa;
    border-radius: 6px;
    padding: 0.8rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
h1 {
    color: #1e3a5f;
    font-size: 1.6rem;
    font-weight: 500;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 4px;
    padding: 0.4rem 0.8rem;
}
.stTabs [aria-selected="true"] {
    background-color: #2e86c1;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'selected_markets' not in st.session_state:
    st.session_state.selected_markets = []
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = {}
if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.now().date() - timedelta(days=365)
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime.now().date()
if 'total_selected' not in st.session_state:
    st.session_state.total_selected = 0

# Title
st.title("Global Equity Analytics")

# Sidebar
st.sidebar.header("Settings")

# Load config
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except:
    st.error("config.yaml not found")
    st.stop()

# Market selection
markets = list(config.get('indices', {}).keys())
selected_markets = st.sidebar.multiselect(
    "Markets",
    options=markets,
    default=markets[:2] if len(markets) >= 2 else markets
)

# Date range
end_date = datetime.now().date()
start_date = end_date - timedelta(days=365)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start", start_date, max_value=end_date)
with col2:
    end_date = st.date_input("End", end_date, max_value=end_date)

if start_date >= end_date:
    st.sidebar.error("End date must be after start date")

# Stock selection
st.sidebar.subheader("Stocks")
selected_tickers = {}
total = 0

for market in selected_markets:
    market_config = config['indices'].get(market, {})
    tickers = market_config.get('tickers', [])
    
    if tickers:
        with st.sidebar.expander(market):
            selected = st.multiselect(
                f"Select",
                options=tickers,
                default=tickers[:2] if len(tickers) >= 2 else tickers,
                key=f"{market}"
            )
            if selected:
                selected_tickers[market] = selected
                total += len(selected)

# Run button
if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
    st.session_state.run_analysis = True
    st.session_state.selected_markets = selected_markets
    st.session_state.selected_tickers = selected_tickers
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    st.session_state.total_selected = total

# Helper functions
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sharpe(returns, rf=0.02):
    if returns.empty or returns.std() == 0:
        return 0
    excess = returns - rf / 252
    return excess.mean() / returns.std() * (252 ** 0.5)

def calculate_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    return dd

# Main content
if st.session_state.run_analysis:
    
    markets = st.session_state.selected_markets
    tickers_dict = st.session_state.selected_tickers
    start = st.session_state.start_date
    end = st.session_state.end_date
    total = st.session_state.total_selected
    
    if total == 0:
        st.warning("Please select at least one stock")
        st.stop()
    
    st.info(f"Loading {total} stocks...")
    
    all_prices = {}
    all_data = {}
    
    # Convert dates for yfinance
    tz = pytz.UTC
    start_aware = tz.localize(datetime.combine(start, datetime.min.time()))
    end_aware = tz.localize(datetime.combine(end, datetime.min.time()))
    
    for market in markets:
        if market in tickers_dict:
            for ticker in tickers_dict[market]:
                try:
                    import yfinance as yf
                    
                    data = yf.download(
                        ticker,
                        start=start_aware,
                        end=end_aware,
                        progress=False,
                        auto_adjust=True
                    )
                    
                    if not data.empty and 'Close' in data.columns:
                        # Remove timezone
                        if data.index.tz is not None:
                            data.index = data.index.tz_localize(None)
                        
                        all_data[ticker] = data
                        
                        if market not in all_prices:
                            all_prices[market] = {}
                        all_prices[market][ticker] = data['Close']
                        
                except Exception as e:
                    st.warning(f"Could not load {ticker}")
    
    if not all_prices:
        st.error("No data loaded")
        st.stop()
    
    # Create price dataframes
    for market in all_prices:
        if all_prices[market]:
            all_prices[market] = pd.DataFrame(all_prices[market])
    
    # Combine all prices
    combined = []
    for df in all_prices.values():
        if not df.empty:
            combined.append(df)
    
    if combined:
        prices = pd.concat(combined, axis=1).dropna(axis=1, how='all')
    else:
        prices = pd.DataFrame()
    
    if not prices.empty:
        returns = prices.pct_change().dropna()
        portfolio_returns = returns.mean(axis=1)
    else:
        returns = pd.DataFrame()
        portfolio_returns = pd.Series()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Technical", "Portfolio", "Risk"])
    
    # Tab 1: Overview
    with tab1:
        st.header("Market Performance")
        
        if not prices.empty:
            normalized = prices / prices.iloc[0] * 100
            fig = px.line(normalized, title="Normalized Prices")
            fig.update_layout(height=450, template="simple_white")
            st.plotly_chart(fig, use_container_width=True)
        
        if len(prices.columns) > 1 and not returns.empty:
            st.subheader("Correlation")
            corr = returns.corr()
            fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale="RdBu", zmin=-1, zmax=1)
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("Summary")
        summary = []
        for market, df in all_prices.items():
            if not df.empty:
                summary.append({
                    "Market": market,
                    "Stocks": len(df.columns),
                    "Latest": f"{df.iloc[-1].mean():.2f}"
                })
        if summary:
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
    
    # Tab 2: Technical
    with tab2:
        st.header("Technical Analysis")
        
        if all_data:
            ticker_list = list(all_data.keys())
            selected = st.selectbox("Stock", ticker_list)
            
            if selected and selected in all_data:
                df = all_data[selected].copy()
                
                if not df.empty:
                    df['SMA20'] = df['Close'].rolling(20).mean()
                    df['SMA50'] = df['Close'].rolling(50).mean()
                    df['RSI'] = calculate_rsi(df['Close'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name="SMA20"))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name="SMA50"))
                    fig.update_layout(title=selected, height=450, template="simple_white")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(title="RSI (14)", height=300, template="simple_white")
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current", f"{df['Close'].iloc[-1]:.2f}")
                    c2.metric("SMA20", f"{df['SMA20'].iloc[-1]:.2f}")
                    c3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    
    # Tab 3: Portfolio
    with tab3:
        st.header("Portfolio Optimization")
        
        if len(prices.columns) >= 2:
            weights = {ticker: 1/len(prices.columns) for ticker in prices.columns}
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Assets", len(prices.columns))
            
            if not portfolio_returns.empty:
                ret = (1 + portfolio_returns).prod() - 1
                c2.metric("Return", f"{ret:.2%}")
                sharpe = calculate_sharpe(portfolio_returns)
                c3.metric("Sharpe", f"{sharpe:.3f}")
            
            weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=["Weight"])
            weights_df = weights_df.sort_values("Weight", ascending=False)
            
            fig = px.bar(weights_df, x=weights_df.index, y="Weight", title="Equal Weight Portfolio")
            fig.update_yaxis(tickformat=".0%")
            fig.update_layout(height=450, template="simple_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least 2 stocks for portfolio analysis")
    
    # Tab 4: Risk
    with tab4:
        st.header("Risk Analytics")
        
        if not portfolio_returns.empty:
            total_ret = (1 + portfolio_returns).prod() - 1
            vol = portfolio_returns.std() * (252 ** 0.5)
            sharpe = calculate_sharpe(portfolio_returns)
            dd = calculate_drawdown(portfolio_returns)
            max_dd = dd.min()
            var_95 = portfolio_returns.quantile(0.05)
            win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Return", f"{total_ret:.2%}")
            c2.metric("Volatility", f"{vol:.2%}")
            c3.metric("Sharpe", f"{sharpe:.3f}")
            c4.metric("Max DD", f"{max_dd:.2%}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VaR (95%)", f"{var_95:.2%}")
            c2.metric("Win Rate", f"{win_rate:.2%}")
            
            # Drawdown chart
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", name="Drawdown"))
            fig_dd.update_layout(title="Drawdown", yaxis_title="%", height=350, template="simple_white")
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Returns distribution
            fig_hist = px.histogram(portfolio_returns, nbins=50, title="Daily Returns")
            fig_hist.update_xaxis(tickformat=".2%")
            fig_hist.update_layout(height=350, template="simple_white")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Insufficient data for risk analysis")

else:
    # Welcome screen
    st.info("Select markets and stocks from the sidebar, then click Run Analysis")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("Markets")
        st.markdown("- USA\n- Turkey\n- Germany\n- France\n- Japan\n- Australia")
    
    with c2:
        st.subheader("Analysis")
        st.markdown("- Performance\n- Technical Indicators\n- Risk Metrics\n- Drawdown")
    
    with c3:
        st.subheader("Features")
        st.markdown("- Correlation\n- Sharpe Ratio\n- VaR\n- RSI")

st.markdown("---")
st.caption("Data from Yahoo Finance")
