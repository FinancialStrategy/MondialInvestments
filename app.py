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

def calculate_atr(high, low, close, period=14):
    """Calculate ATR without TA-Lib"""
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_drawdown(returns):
    """Calculate drawdown series"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    sharpe = (excess_returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate Sortino ratio"""
    excess_returns = returns - risk_free_rate / 252
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() if len(negative_returns) > 0 else returns.std()
    sortino = (excess_returns.mean() / downside_std * np.sqrt(252)) if downside_std != 0 else 0
    return sortino

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
    default=available_markets[:3] if len(available_markets) >= 3 else available_markets
)

# Date range
end_date = datetime.now().date()
start_date = end_date - timedelta(days=730)

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
                default=market_tickers[:4] if len(market_tickers) >= 4 else market_tickers,
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
    from modules.data_loader import fetch_market_data
    
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
    all_returns = {}
    progress_bar = st.progress(0)
    
    for idx, market in enumerate(selected_markets):
        if market in selected_tickers_dict:
            tickers = selected_tickers_dict[market]
            progress_bar.progress((idx + 0.5) / len(selected_markets))
            
            ticker_data = fetch_market_data(tickers, str(start_date), str(end_date))
            
            if ticker_data:
                close_prices = {}
                for ticker, df in ticker_data.items():
                    if 'Close' in df.columns:
                        close_prices[ticker] = df['Close']
                        all_ohlc_data[ticker] = df
                
                if close_prices:
                    prices_df = pd.DataFrame(close_prices)
                    if not prices_df.empty:
                        all_prices[market] = prices_df
                        all_returns[market] = prices_df.pct_change().dropna()
    
    progress_bar.progress(1.0)
    progress_bar.empty()
    
    if not all_prices:
        st.error("❌ No data loaded. Please check your selections.")
        st.stop()
    
    # Combine all data
    combined_prices = pd.concat(all_prices.values(), axis=1).dropna(axis=1, how='all')
    combined_returns = combined_prices.pct_change().dropna()
    
    # Equal weight portfolio returns
    if not combined_returns.empty and len(combined_returns.columns) > 0:
        portfolio_returns = combined_returns.mean(axis=1)
    else:
        portfolio_returns = pd.Series()
    
    # ==================== TABS ====================
    tabs_list = ["📊 Market Overview", "📈 Technical Analysis", "🎯 Portfolio Optimization", "⚠️ Risk Analytics"]
    if supertrend_params['enabled']:
        tabs_list.append("🟢 Supertrend Strategy")
    
    tabs = st.tabs(tabs_list)
    
    # ==================== TAB 1: MARKET OVERVIEW ====================
    with tabs[0]:
        st.header("Market Performance")
        
        # Normalized price chart
        if not combined_prices.empty:
            normalized = combined_prices / combined_prices.iloc[0] * 100
            
            fig = px.line(
                normalized,
                title="Normalized Price Performance (Base 100)",
                labels={"value": "Price", "variable": "Ticker", "index": "Date"}
            )
            fig.update_layout(height=500, template='plotly_white', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        if len(combined_prices.columns) > 1:
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
        
        for market, prices in all_prices.items():
            if not prices.empty:
                rets = prices.pct_change().dropna()
                avg_return = rets.mean().mean()
                avg_vol = rets.std().mean()
                sharpe = (avg_return / avg_vol * np.sqrt(252)) if avg_vol > 0 else 0
                
                summary_data.append({
                    'Market': market,
                    'Stocks': len(prices.columns),
                    'Avg Return': f"{avg_return:.4%}",
                    'Avg Volatility': f"{avg_vol:.4%}",
                    'Avg Sharpe': f"{sharpe:.3f}"
                })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # Returns distribution
        if not combined_returns.empty:
            st.subheader("Returns Distribution")
            fig_hist = px.histogram(
                combined_returns.stack().reset_index(drop=True),
                nbins=50,
                title="Daily Returns Distribution",
                labels={"value": "Return", "count": "Frequency"}
            )
            fig_hist.update_layout(height=400, template='plotly_white')
            fig_hist.update_xaxis(tickformat=".2%")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # ==================== TAB 2: TECHNICAL ANALYSIS ====================
    with tabs[1]:
        st.header("Technical Analysis")
        
        if all_ohlc_data:
            all_tickers = list(all_ohlc_data.keys())
            selected_ticker = st.selectbox("Select Stock for Analysis", all_tickers)
            
            if selected_ticker and selected_ticker in all_ohlc_data:
                df = all_ohlc_data[selected_ticker].copy()
                
                if not df.empty and 'Close' in df.columns:
                    # Calculate indicators
                    df['SMA_20'] = df['Close'].rolling(20).mean()
                    df['SMA_50'] = df['Close'].rolling(50).mean()
                    df['SMA_200'] = df['Close'].rolling(200).mean()
                    df['RSI'] = calculate_rsi(df['Close'])
                    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
                    
                    macd, signal, hist = calculate_macd(df['Close'])
                    df['MACD'] = macd
                    df['MACD_Signal'] = signal
                    df['MACD_Hist'] = hist
                    
                    upper, middle, lower = calculate_bollinger_bands(df['Close'])
                    df['BB_Upper'] = upper
                    df['BB_Middle'] = middle
                    df['BB_Lower'] = lower
                    
                    # Create subplot figure
                    fig = make_subplots(
                        rows=4, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.4, 0.2, 0.2, 0.2],
                        subplot_titles=(f"{selected_ticker} - Price", "RSI (14)", "MACD", "Bollinger Bands")
                    )
                    
                    # Price chart with MAs
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='red')), row=1, col=1)
                    
                    # RSI
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                    
                    # MACD
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')), row=3, col=1)
                    
                    colors = ['green' if v >= 0 else 'red' for v in df['MACD_Hist']]
                    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=colors), row=3, col=1)
                    
                    # Bollinger Bands
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')), row=4, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')), row=4, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')), row=4, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Middle', line=dict(color='orange')), row=4, col=1)
                    
                    fig.update_layout(height=900, template='plotly_white', showlegend=True)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
                    fig.update_yaxes(title_text="MACD", row=3, col=1)
                    fig.update_yaxes(title_text="Price", row=4, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Current indicator values
                    st.subheader("Current Indicator Values")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
                    with col2:
                        st.metric("SMA 20", f"{df['SMA_20'].iloc[-1]:.2f}")
                    with col3:
                        st.metric("SMA 50", f"{df['SMA_50'].iloc[-1]:.2f}")
                    with col4:
                        st.metric("MACD", f"{df['MACD'].iloc[-1]:.3f}")
    
    # ==================== TAB 3: PORTFOLIO OPTIMIZATION ====================
    with tabs[2]:
        st.header("Portfolio Optimization")
        
        if len(combined_prices.columns) >= 2:
            try:
                from modules.portfolio_optimizer import PortfolioOptimizer
                
                optimizer = PortfolioOptimizer(combined_prices)
                
                # Strategy selection
                strategy = st.selectbox(
                    "Optimization Strategy",
                    ["Max Sharpe Ratio", "Min Volatility", "Max Quadratic Utility"]
                )
                
                if strategy == "Max Sharpe Ratio":
                    result = optimizer.optimize_max_sharpe()
                elif strategy == "Min Volatility":
                    result = optimizer.optimize_min_volatility()
                else:
                    risk_aversion = st.slider("Risk Aversion", 1.0, 10.0, 3.0)
                    result = optimizer.optimize_max_quadratic_utility(risk_aversion)
                
                if result['status'] == 'success' and result['weights']:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Expected Return", f"{result['expected_return']:.2%}")
                    with col2:
                        st.metric("Volatility", f"{result['volatility']:.2%}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
                    
                    # Weights chart
                    weights_df = pd.DataFrame.from_dict(result['weights'], orient='index', columns=['Weight'])
                    weights_df = weights_df[weights_df['Weight'] > 0.01].sort_values('Weight', ascending=False)
                    
                    if not weights_df.empty:
                        fig = px.bar(
                            weights_df,
                            x=weights_df.index,
                            y='Weight',
                            title="Optimal Portfolio Weights",
                            color='Weight',
                            color_continuous_scale='Viridis',
                            text=weights_df['Weight'].apply(lambda x: f'{x:.1%}')
                        )
                        fig.update_layout(height=500, template='plotly_white')
                        fig.update_yaxis(tickformat='.0%')
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download button
                        csv = weights_df.to_csv().encode('utf-8')
                        st.download_button(
                            label="📥 Download Weights",
                            data=csv,
                            file_name='portfolio_weights.csv',
                            mime='text/csv'
                        )
                else:
                    st.warning("Optimization failed. Try different assets.")
            except ImportError:
                st.error("Portfolio optimizer module not available")
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
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe = calculate_sharpe_ratio(portfolio_returns)
            sortino = calculate_sortino_ratio(portfolio_returns)
            drawdown = calculate_drawdown(portfolio_returns)
            max_drawdown = drawdown.min()
            var_95 = portfolio_returns.quantile(0.05)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
            
            # Display metrics
            st.subheader("Portfolio Risk Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{total_return:.2%}")
                st.metric("Annual Return", f"{annual_return:.2%}")
            with col2:
                st.metric("Volatility", f"{volatility:.2%}")
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
                st.metric("Sortino Ratio", f"{sortino:.3f}")
            with col4:
                st.metric("VaR (95%)", f"{var_95:.2%}")
                st.metric("CVaR (95%)", f"{cvar_95:.2%}")
                st.metric("Win Rate", f"{win_rate:.2%}")
            
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
            
            # Rolling metrics
            st.subheader("Rolling Metrics (60-day window)")
            
            rolling_sharpe = portfolio_returns.rolling(60).apply(
                lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() != 0 else 0
            )
            rolling_vol = portfolio_returns.rolling(60).std() * np.sqrt(252)
            
            fig_roll = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig_roll.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name='Sharpe'), row=1, col=1)
            fig_roll.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol, name='Volatility'), row=2, col=1)
            fig_roll.update_layout(height=500, template='plotly_white')
            fig_roll.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
            fig_roll.update_yaxes(title_text="Volatility", tickformat=".2%", row=2, col=1)
            st.plotly_chart(fig_roll, use_container_width=True)
            
            # Monthly returns heatmap
            st.subheader("Monthly Returns")
            monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
            if not monthly_returns.empty:
                monthly_pivot = pd.DataFrame({
                    'Year': monthly_returns.index.year,
                    'Month': monthly_returns.index.month,
                    'Return': monthly_returns.values
                }).pivot(index='Year', columns='Month', values='Return')
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_pivot.columns = month_names[:len(monthly_pivot.columns)]
                
                fig_heat = px.imshow(
                    monthly_pivot * 100,
                    text_auto='.1f',
                    aspect="auto",
                    color_continuous_scale="RdYlGn",
                    title="Monthly Returns (%)",
                    zmin=-10, zmax=10
                )
                fig_heat.update_layout(height=500)
                st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.warning("Insufficient data for risk analysis")
    
    # ==================== TAB 5: SUPERTREND STRATEGY ====================
    if supertrend_params['enabled'] and len(tabs) > 4:
        with tabs[4]:
            st.header("Supertrend Trading Strategy")
            
            try:
                from modules.supertrend_signals import SupertrendAnalyzer
                
                if all_ohlc_data:
                    all_tickers = list(all_ohlc_data.keys())
                    selected = st.selectbox("Select Stock for Supertrend Analysis", all_tickers, key="st_selector")
                    
                    if selected and selected in all_ohlc_data:
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
                else:
                    st.info("No stock data available for Supertrend analysis")
            except ImportError:
                st.error("Supertrend module not available")
            except Exception as e:
                st.error(f"Supertrend error: {e}")

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
        - Portfolio Optimization (PyPortfolioOpt)
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
        - Download Reports
        """)
    
    st.markdown("---")
    st.caption("Select markets and stocks from the sidebar to begin analysis.")

# Footer
st.markdown("---")
st.caption("Global Equity Analytics Platform | Data from Yahoo Finance")
