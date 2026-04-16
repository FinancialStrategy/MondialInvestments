"""
Global Equity Analytics Platform
Comprehensive Multi-Market Portfolio Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Global Equity Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple custom CSS
st.markdown("""
<style>
.stMetric {
    background-color: #F8F9FA;
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
h1 {
    color: #1E3A5F;
    font-size: 1.8rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    background-color: #2E86C1;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'run_analysis' not in st.session_state:
    st.session_state['run_analysis'] = False
if 'selected_markets' not in st.session_state:
    st.session_state['selected_markets'] = []
if 'selected_tickers' not in st.session_state:
    st.session_state['selected_tickers'] = {}
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = datetime.now().date() - timedelta(days=730)
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = datetime.now().date()
if 'total_selected' not in st.session_state:
    st.session_state['total_selected'] = 0
if 'supertrend_params' not in st.session_state:
    st.session_state['supertrend_params'] = {
        'enabled': True,
        'period': 10,
        'multiplier': 3.0
    }

# Title
st.title("📊 Global Equity Analytics Platform")
st.markdown("---")

# ==================== HELPER FUNCTIONS ====================

def calculate_rsi(prices, period=14):
    """Calculate RSI without TA-Lib"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD without TA-Lib"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands without TA-Lib"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_drawdown(returns):
    """Calculate drawdown series"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio"""
    if returns.empty or returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate / 252
    sharpe = (excess_returns.mean() / returns.std() * np.sqrt(252))
    return sharpe

# ==================== SIDEBAR ====================

st.sidebar.header("⚙️ Configuration")

# Load config
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("❌ config.yaml file not found")
    st.stop()

# Market selection
available_markets = list(config.get('indices', {}).keys())
if not available_markets:
    st.error("No markets found in config.yaml")
    st.stop()

selected_markets = st.sidebar.multiselect(
    "🌍 Select Markets",
    options=available_markets,
    default=available_markets[:2] if len(available_markets) >= 2 else available_markets
)

# Date range
end_date = datetime.now().date()
start_date = end_date - timedelta(days=365)  # 1 year for better performance

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("📅 Start Date", start_date, max_value=end_date)
with col2:
    end_date = st.date_input("📅 End Date", end_date, max_value=end_date)

if start_date >= end_date:
    st.sidebar.error("End date must be after start date")

# Stock selection
st.sidebar.subheader("📊 Stock Selection")
selected_tickers = {}
total_selected = 0

for market in selected_markets:
    market_config = config['indices'].get(market, {})
    market_tickers = market_config.get('tickers', [])
    currency = market_config.get('currency', 'USD')
    
    if market_tickers:
        with st.sidebar.expander(f"{market} ({currency})"):
            selected = st.multiselect(
                f"Select stocks",
                options=market_tickers,
                default=market_tickers[:2] if len(market_tickers) >= 2 else market_tickers,
                key=f"{market}_selector"
            )
            if selected:
                selected_tickers[market] = selected
                total_selected += len(selected)

# Supertrend settings
st.sidebar.markdown("---")
st.sidebar.subheader("🟢 Supertrend Strategy")
enable_supertrend = st.sidebar.checkbox("Enable Supertrend", value=True)

if enable_supertrend:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        supertrend_period = st.slider("ATR Period", 5, 20, 10)
    with col2:
        supertrend_multiplier = st.slider("ATR Multiplier", 1.0, 5.0, 3.0, 0.5)

# Run button
if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
    st.session_state['run_analysis'] = True
    st.session_state['selected_markets'] = selected_markets
    st.session_state['selected_tickers'] = selected_tickers
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    st.session_state['total_selected'] = total_selected
    st.session_state['supertrend_params'] = {
        'enabled': enable_supertrend,
        'period': supertrend_period if enable_supertrend else 10,
        'multiplier': supertrend_multiplier if enable_supertrend else 3.0
    }

# ==================== MAIN CONTENT ====================

if st.session_state.get('run_analysis', False):
    
    selected_markets = st.session_state['selected_markets']
    selected_tickers_dict = st.session_state['selected_tickers']
    start_date = st.session_state['start_date']
    end_date = st.session_state['end_date']
    total_selected = st.session_state['total_selected']
    supertrend_params = st.session_state['supertrend_params']
    
    if total_selected == 0:
        st.warning("⚠️ Please select at least one stock")
        st.stop()
    
    # ==================== DATA LOADING ====================
    st.info(f"📥 Loading data for {total_selected} stocks...")
    
    all_prices = {}
    all_ohlc_data = {}
    
    for market in selected_markets:
        if market in selected_tickers_dict:
            tickers = selected_tickers_dict[market]
            
            for ticker in tickers:
                try:
                    import yfinance as yf
                    
                    # Download data with error handling
                    data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=True
                    )
                    
                    if not data.empty and 'Close' in data.columns:
                        # Store close prices
                        if market not in all_prices:
                            all_prices[market] = {}
                        all_prices[market][ticker] = data['Close']
                        
                        # Store OHLC data
                        all_ohlc_data[ticker] = data
                        
                except Exception as e:
                    st.warning(f"Could not load {ticker}: {e}")
    
    if not all_prices:
        st.error("❌ No data loaded. Please check your selections.")
        st.stop()
    
    # Convert to DataFrames
    for market in all_prices:
        if all_prices[market]:
            all_prices[market] = pd.DataFrame(all_prices[market])
    
    # Combine all prices
    combined_prices = []
    for market, df in all_prices.items():
        if not df.empty:
            combined_prices.append(df)
    
    if combined_prices:
        combined_prices = pd.concat(combined_prices, axis=1).dropna(axis=1, how='all')
    else:
        combined_prices = pd.DataFrame()
    
    if not combined_prices.empty:
        combined_returns = combined_prices.pct_change().dropna()
        portfolio_returns = combined_returns.mean(axis=1) if not combined_returns.empty else pd.Series()
    else:
        combined_returns = pd.DataFrame()
        portfolio_returns = pd.Series()
    
    # ==================== TABS ====================
    tabs_list = ["📊 Market Overview", "📈 Technical Analysis", "🎯 Portfolio Optimization", "⚠️ Risk Analytics"]
    if supertrend_params['enabled']:
        tabs_list.append("🟢 Supertrend Strategy")
    
    tabs = st.tabs(tabs_list)
    
    # ==================== TAB 1: MARKET OVERVIEW ====================
    with tabs[0]:
        st.header("Market Performance")
        
        if not combined_prices.empty:
            # Normalized price chart
            normalized = combined_prices / combined_prices.iloc[0] * 100
            
            fig = px.line(
                normalized,
                title="Normalized Price Performance (Base 100)",
                labels={"value": "Price", "variable": "Ticker", "index": "Date"}
            )
            fig.update_layout(height=500, template='plotly_white', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        if len(combined_prices.columns) > 1 and not combined_returns.empty:
            st.subheader("Asset Correlation Matrix")
            corr_matrix = combined_returns.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Correlation Heatmap",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Market summary
        st.subheader("Market Summary")
        summary_data = []
        
        for market, df in all_prices.items():
            if not df.empty:
                rets = df.pct_change().dropna()
                if not rets.empty:
                    summary_data.append({
                        'Market': market,
                        'Stocks': len(df.columns),
                        'Latest Close': f"{df.iloc[-1].mean():.2f}",
                        'Avg Daily Return': f"{rets.mean().mean():.4%}"
                    })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # ==================== TAB 2: TECHNICAL ANALYSIS ====================
    with tabs[1]:
        st.header("Technical Analysis")
        
        if all_ohlc_data:
            all_tickers = list(all_ohlc_data.keys())
            if all_tickers:
                selected_ticker = st.selectbox("Select Stock for Analysis", all_tickers)
                
                if selected_ticker and selected_ticker in all_ohlc_data:
                    df = all_ohlc_data[selected_ticker].copy()
                    
                    if not df.empty and 'Close' in df.columns:
                        # Calculate indicators
                        df['SMA_20'] = df['Close'].rolling(20).mean()
                        df['SMA_50'] = df['Close'].rolling(50).mean()
                        df['RSI'] = calculate_rsi(df['Close'])
                        
                        # Price chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')))
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red')))
                        
                        fig.update_layout(
                            title=f"{selected_ticker} - Price & Moving Averages",
                            yaxis_title="Price",
                            height=500,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # RSI chart
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        fig_rsi.update_layout(title="RSI (14)", yaxis_title="RSI", height=400, template='plotly_white')
                        st.plotly_chart(fig_rsi, use_container_width=True)
                        
                        # Current values
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"{df['Close'].iloc[-1]:.2f}")
                        with col2:
                            st.metric("SMA 20", f"{df['SMA_20'].iloc[-1]:.2f}")
                        with col3:
                            st.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
            else:
                st.warning("No technical data available")
    
    # ==================== TAB 3: PORTFOLIO OPTIMIZATION ====================
    with tabs[2]:
        st.header("Portfolio Optimization")
        
        if len(combined_prices.columns) >= 2:
            try:
                # Simple equal weight as fallback
                st.subheader("Equal Weight Portfolio")
                
                weights = {ticker: 1/len(combined_prices.columns) for ticker in combined_prices.columns}
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Assets", len(combined_prices.columns))
                with col2:
                    if not portfolio_returns.empty:
                        ret = (1 + portfolio_returns).prod() - 1
                        st.metric("Portfolio Return", f"{ret:.2%}")
                with col3:
                    if not portfolio_returns.empty and portfolio_returns.std() != 0:
                        sharpe = calculate_sharpe_ratio(portfolio_returns)
                        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
                
                # Weights chart
                weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
                weights_df = weights_df.sort_values('Weight', ascending=False)
                
                fig = px.bar(
                    weights_df,
                    x=weights_df.index,
                    y='Weight',
                    title="Portfolio Weights",
                    color='Weight',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=500, template='plotly_white')
                fig.update_yaxis(tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Optimization error: {e}")
        else:
            st.warning("Select at least 2 assets for portfolio optimization")
    
    # ==================== TAB 4: RISK ANALYTICS ====================
    with tabs[3]:
        st.header("Risk Analytics")
        
        if not portfolio_returns.empty:
            # Calculate metrics
            total_return = (1 + portfolio_returns).prod() - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe = calculate_sharpe_ratio(portfolio_returns)
            drawdown = calculate_drawdown(portfolio_returns)
            max_drawdown = drawdown.min()
            var_95 = portfolio_returns.quantile(0.05)
            win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{total_return:.2%}")
            with col2:
                st.metric("Volatility", f"{volatility:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            with col4:
                st.metric("Win Rate", f"{win_rate:.2%}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
            with col2:
                st.metric("VaR (95%)", f"{var_95:.2%}")
            
            # Drawdown chart
            st.subheader("Drawdown Analysis")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='red', width=1),
                fillcolor='rgba(255,0,0,0.3)'
            ))
            fig_dd.update_layout(height=400, template='plotly_white', yaxis_title="Drawdown (%)")
            fig_dd.add_hline(y=-10, line_dash="dash", line_color="orange")
            fig_dd.add_hline(y=-20, line_dash="dash", line_color="red")
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Returns distribution
            st.subheader("Returns Distribution")
            fig_hist = px.histogram(
                portfolio_returns,
                nbins=50,
                title="Daily Returns Distribution",
                labels={"value": "Return", "count": "Frequency"}
            )
            fig_hist.update_layout(height=400, template='plotly_white')
            fig_hist.update_xaxis(tickformat=".2%")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Insufficient data for risk analysis")
    
    # ==================== TAB 5: SUPERTREND STRATEGY ====================
    if supertrend_params['enabled'] and len(tabs) > 4:
        with tabs[4]:
            st.header("Supertrend Trading Strategy")
            
            if all_ohlc_data:
                all_tickers = list(all_ohlc_data.keys())
                if all_tickers:
                    selected = st.selectbox("Select Stock for Supertrend Analysis", all_tickers, key="st_selector")
                    
                    if selected and selected in all_ohlc_data:
                        try:
                            from modules.supertrend_signals import SupertrendAnalyzer
                            
                            df = all_ohlc_data[selected].copy()
                            
                            if len(df) > supertrend_params['period']:
                                analyzer = SupertrendAnalyzer(
                                    period=supertrend_params['period'],
                                    multiplier=supertrend_params['multiplier']
                                )
                                signals = analyzer.generate_signals(df)
                                
                                # Current status
                                current_trend = signals['Trend'].iloc[-1]
                                current_price = signals['Close'].iloc[-1]
                                current_supertrend = signals['Supertrend'].iloc[-1]
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    trend_text = "🟢 UPTREND" if current_trend == 1 else "🔴 DOWNTREND"
                                    st.metric("Current Trend", trend_text)
                                with col2:
                                    st.metric("Current Price", f"{current_price:.2f}")
                                with col3:
                                    st.metric("Supertrend", f"{current_supertrend:.2f}")
                                with col4:
                                    last_signal = signals[signals['Signal'] != 0]
                                    if not last_signal.empty:
                                        sig = last_signal.iloc[-1]['Signal']
                                        signal_text = "BUY" if sig == 1 else "SELL"
                                        st.metric("Last Signal", signal_text)
                                    else:
                                        st.metric("Last Signal", "None")
                                
                                # Chart
                                fig = analyzer.create_signal_chart(f"{selected}")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Performance
                                analyzer.create_performance_dashboard()
                            else:
                                st.warning(f"Insufficient data. Need at least {supertrend_params['period']} days.")
                        except ImportError:
                            st.error("Supertrend module not available")
                        except Exception as e:
                            st.error(f"Supertrend error: {e}")
                else:
                    st.info("No stock data available for Supertrend analysis")

else:
    # Welcome screen
    st.info("👈 Select markets and stocks from the sidebar, then click 'Run Analysis'")
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌍 Global Coverage")
        st.markdown("""
        - 🇹🇷 Turkey (BIST)
        - 🇺🇸 United States
        - 🇩🇪 Germany (DAX)
        - 🇫🇷 France (CAC)
        - 🇮🇹 Italy (FTSE MIB)
        - 🇨🇭 Switzerland (SMI)
        - 🇯🇵 Japan (Nikkei)
        - 🇦🇺 Australia (ASX)
        """)
    
    with col2:
        st.subheader("📊 Features")
        st.markdown("""
        - Market Overview & Correlation
        - Technical Indicators (SMA, EMA, RSI, MACD, Bollinger)
        - Portfolio Optimization
        - Risk Analytics (Sharpe, Sortino, VaR, Drawdown)
        - Supertrend Trading Strategy
        """)
    
    with col3:
        st.subheader("🎯 Analysis Tools")
        st.markdown("""
        - Interactive Charts
        - Performance Metrics
        - Risk Assessment
        - Signal Generation
        """)
    
    st.markdown("---")
    st.caption("Global Equity Analytics Platform | Data from Yahoo Finance")

# Footer
st.markdown("---")
st.caption("Select markets and stocks from the sidebar to begin analysis.")
