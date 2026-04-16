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

# Page configuration MUST be first Streamlit command
st.set_page_config(
    page_title="Global Equity Analytics Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
try:
    with open('assets/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("""
    <style>
    .stMetric { background-color: #F8F9FA; border-radius: 10px; padding: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🌍 Global Equity Analytics Platform")
st.markdown("""
*Professional Multi-Market Portfolio Analysis | Powered by FinQuant, PyPortfolioOpt, QuantStats & TA-Lib*
""")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Load config
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("❌ config.yaml file not found! Please ensure the file exists.")
    st.stop()

# Market selection
available_markets = list(config.get('indices', {}).keys())
if not available_markets:
    st.error("❌ No markets found in config.yaml!")
    st.stop()

selected_markets = st.sidebar.multiselect(
    "🌍 Select Global Markets",
    options=available_markets,
    default=available_markets[:3] if len(available_markets) >= 3 else available_markets
)

# Date range
end_date = datetime.now().date()
default_days = config.get('settings', {}).get('default_period_days', 730)
start_date = end_date - timedelta(days=default_days)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("📅 Start Date", start_date, max_value=end_date)
with col2:
    end_date = st.date_input("📅 End Date", end_date, max_value=end_date)

# Validate date range
if start_date >= end_date:
    st.sidebar.error("❌ End date must be after start date!")

# Ticker selection per market
st.sidebar.subheader("📊 Stock Selection")
selected_tickers = {}
total_selected = 0

for market in selected_markets:
    market_config = config['indices'].get(market, {})
    market_tickers = market_config.get('tickers', [])
    
    if market_tickers:
        with st.sidebar.expander(f"{market} Market ({market_config.get('currency', 'USD')})"):
            selected = st.multiselect(
                f"Select {market} stocks",
                options=market_tickers,
                default=market_tickers[:5] if len(market_tickers) >= 5 else market_tickers,
                key=f"{market}_selector"
            )
            if selected:
                selected_tickers[market] = selected
                total_selected += len(selected)

# Supertrend Configuration in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🟢 Supertrend Strategy")
enable_supertrend = st.sidebar.checkbox("Enable Supertrend Analysis", value=True)

if enable_supertrend:
    st.sidebar.markdown("**Indicator Parameters:**")
    supertrend_period = st.sidebar.slider("ATR Period", 5, 20, 10, 
                                         help="ATR hesaplama periyodu - Düşük değer daha hassas")
    supertrend_multiplier = st.sidebar.slider("ATR Multiplier", 1.0, 5.0, 3.0, 0.5,
                                             help="ATR çarpanı - Yüksek değer daha az sinyal")
    
    st.sidebar.markdown("**Scan Settings:**")
    scan_all_stocks = st.sidebar.checkbox("Scan All Selected Stocks", value=True)

# Run button
if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
    st.session_state['run_analysis'] = True
    st.session_state['selected_markets'] = selected_markets
    st.session_state['selected_tickers'] = selected_tickers
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    st.session_state['supertrend_params'] = {
        'enabled': enable_supertrend,
        'period': supertrend_period if enable_supertrend else 10,
        'multiplier': supertrend_multiplier if enable_supertrend else 3.0,
        'scan_all': scan_all_stocks if enable_supertrend else False
    }

# Main content area
if st.session_state.get('run_analysis', False):
    # Import modules
    from modules.data_loader import fetch_market_data, get_benchmark_data, load_config
    from modules.technical_indicators import add_technical_indicators, create_candlestick_with_indicators
    
    # Import portfolio modules
    from modules.portfolio_optimizer import PortfolioOptimizer
    from modules.risk_metrics import RiskAnalyzer
    
    # Import quantstats
    import quantstats as qs
    
    # Get parameters
    selected_markets = st.session_state['selected_markets']
    selected_tickers_dict = st.session_state['selected_tickers']
    start_date = st.session_state['start_date']
    end_date = st.session_state['end_date']
    supertrend_params = st.session_state['supertrend_params']
    
    if total_selected == 0:
        st.warning("⚠️ Please select at least one stock from the sidebar to begin the analysis.")
        st.stop()
    
    # Data loading
    st.info(f"📥 Loading data for {total_selected} stocks across {len(selected_markets)} markets...")
    
    all_prices = {}
    all_returns = {}
    all_ohlc_data = {}  # For Supertrend analysis
    
    progress_bar = st.progress(0)
    
    for idx, market in enumerate(selected_markets):
        if market in selected_tickers_dict:
            tickers = selected_tickers_dict[market]
            progress_bar.progress((idx + 0.5) / len(selected_markets))
            
            # Fetch data
            ticker_data = fetch_market_data(tickers, str(start_date), str(end_date))
            
            if ticker_data:
                # Extract close prices for each ticker
                close_prices = {}
                for ticker, df in ticker_data.items():
                    if 'Close' in df.columns:
                        close_prices[ticker] = df['Close']
                    else:
                        close_prices[ticker] = df.iloc[:, 0] if len(df.columns) > 0 else pd.Series()
                
                # Create DataFrame
                if close_prices:
                    prices_df = pd.DataFrame(close_prices)
                    prices_df = prices_df.dropna(axis=1, how='all')
                    
                    if not prices_df.empty:
                        all_prices[market] = prices_df
                        all_returns[market] = prices_df.pct_change().dropna()
                        
                        # Store OHLC data for Supertrend
                        for ticker, df in ticker_data.items():
                            if ticker in prices_df.columns and not df.empty:
                                all_ohlc_data[ticker] = df
    
    progress_bar.progress(1.0)
    progress_bar.empty()
    
    if not all_prices:
        st.error("❌ No data could be loaded. Please check your tickers and internet connection.")
        st.stop()
    
    # Combine all prices for portfolio analysis
    combined_prices = pd.concat(all_prices.values(), axis=1).dropna(axis=1, how='all')
    combined_returns = combined_prices.pct_change().dropna()
    
    # Create tabs
    tabs_list = ["📊 Market Overview", "📈 Technical Analysis", "🎯 Portfolio Optimization", "⚠️ Risk Analytics"]
    if supertrend_params['enabled']:
        tabs_list.append("🟢 Supertrend Strategy")
    
    tabs = st.tabs(tabs_list)
    
    # ==================== TAB 1: MARKET OVERVIEW ====================
    with tabs[0]:
        st.header("📊 Global Market Performance")
        
        # Normalized price chart
        normalized_prices = combined_prices / combined_prices.iloc[0] * 100
        
        fig = px.line(
            normalized_prices,
            title="Normalized Price Performance (Base 100)",
            labels={"value": "Normalized Price", "variable": "Ticker", "index": "Date"}
        )
        fig.update_layout(height=500, template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("📈 Asset Correlation Heatmap")
        if len(combined_prices.columns) > 1:
            corr_matrix = combined_returns.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Asset Correlation Matrix",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Market summary table
        st.subheader("📋 Market Summary Statistics")
        summary_data = []
        
        for market, prices in all_prices.items():
            if not prices.empty:
                returns = prices.pct_change().dropna()
                avg_return = returns.mean().mean()
                avg_volatility = returns.std().mean()
                sharpe = (avg_return / avg_volatility * np.sqrt(252)) if avg_volatility > 0 else 0
                
                summary_data.append({
                    'Market': market,
                    'Stocks': len(prices.columns),
                    'Avg Daily Return': f"{avg_return:.4%}",
                    'Avg Volatility': f"{avg_volatility:.4%}",
                    'Sharpe Ratio': f"{sharpe:.3f}"
                })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # ==================== TAB 2: TECHNICAL ANALYSIS ====================
    with tabs[1]:
        st.header("📈 Interactive Technical Analysis with TA-Lib")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Select ticker for analysis
            all_tickers_flat = []
            for market, prices in all_prices.items():
                for ticker in prices.columns:
                    all_tickers_flat.append(f"{ticker} ({market})")
            
            selected_asset = st.selectbox("Select Asset for Technical Analysis", all_tickers_flat)
        
        with col2:
            st.subheader("Indicator Settings")
            indicators = {
                'sma': st.checkbox("Simple Moving Averages (20/50/200)", value=True),
                'ema': st.checkbox("Exponential Moving Averages (12/26)", value=False),
                'rsi': st.checkbox("RSI (14)", value=True),
                'macd': st.checkbox("MACD", value=True),
                'bollinger': st.checkbox("Bollinger Bands", value=False),
                'atr': st.checkbox("ATR (14)", value=False)
            }
        
        if selected_asset:
            ticker_name = selected_asset.split(" (")[0]
            
            # Find the OHLC data for this ticker
            if ticker_name in all_ohlc_data:
                ohlc_data = all_ohlc_data[ticker_name]
                
                if not ohlc_data.empty and 'Close' in ohlc_data.columns:
                    # Add indicators
                    ohlc_with_indicators = add_technical_indicators(ohlc_data, indicators)
                    
                    # Create interactive chart
                    fig = create_candlestick_with_indicators(ohlc_with_indicators, ticker_name, indicators)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Insufficient data for {ticker_name}")
            else:
                st.warning(f"No OHLC data available for {ticker_name}")
    
    # ==================== TAB 3: PORTFOLIO OPTIMIZATION ====================
    with tabs[2]:
        st.header("🎯 Portfolio Optimization with PyPortfolioOpt")
        
        if len(combined_prices.columns) >= 2:
            from modules.portfolio_optimizer import PortfolioOptimizer
            
            optimizer = PortfolioOptimizer(combined_prices)
            opt_results = optimizer.create_optimization_dashboard()
            
            # Efficient frontier
            st.subheader("📈 Efficient Frontier")
            ef_points = optimizer.get_efficient_frontier_points()
            
            if not ef_points.empty:
                fig_ef = px.scatter(
                    ef_points, x='volatility', y='return',
                    title='Mean-Variance Efficient Frontier',
                    labels={'volatility': 'Volatility (Risk)', 'return': 'Expected Return'},
                    color_discrete_sequence=['blue']
                )
                fig_ef.update_layout(height=500, template='plotly_white')
                fig_ef.update_xaxes(tickformat=".2%")
                fig_ef.update_yaxes(tickformat=".2%")
                st.plotly_chart(fig_ef, use_container_width=True)
        else:
            st.warning("⚠️ Please select at least 2 assets for portfolio optimization")
    
    # ==================== TAB 4: RISK ANALYTICS ====================
    with tabs[3]:
        st.header("⚠️ Advanced Risk Analytics with QuantStats")
        
        if not combined_returns.empty and len(combined_returns.columns) > 0:
            # Create equal-weight portfolio
            equal_weight_returns = combined_returns.mean(axis=1)
            
            # Get benchmark data
            benchmark_symbol = "^GSPC"  # Default S&P 500
            benchmark_data = get_benchmark_data(benchmark_symbol, str(start_date), str(end_date))
            
            if not benchmark_data.empty:
                benchmark_returns = benchmark_data.pct_change().dropna()
                common_idx = equal_weight_returns.index.intersection(benchmark_returns.index)
                aligned_returns = equal_weight_returns[common_idx]
                aligned_benchmark = benchmark_returns[common_idx]
                
                risk_analyzer = RiskAnalyzer(aligned_returns, aligned_benchmark)
                metrics = risk_analyzer.display_metrics_dashboard()
                
                # Rolling metrics
                st.subheader("📊 Rolling Risk Metrics")
                rolling_fig = risk_analyzer.create_rolling_metrics_chart()
                st.plotly_chart(rolling_fig, use_container_width=True)
                
                # Drawdown chart
                st.subheader("📉 Drawdown Analysis")
                drawdown_fig = risk_analyzer.create_drawdown_chart()
                st.plotly_chart(drawdown_fig, use_container_width=True)
                
                # Monthly returns heatmap
                st.subheader("📅 Monthly Returns Heatmap")
                try:
                    monthly_returns = qs.stats.monthly_returns(aligned_returns)
                    if not monthly_returns.empty and len(monthly_returns) > 0:
                        monthly_pivot = monthly_returns.unstack()
                        
                        fig_monthly = px.imshow(
                            monthly_pivot,
                            text_auto='.2%',
                            aspect="auto",
                            color_continuous_scale="RdYlGn",
                            title="Monthly Returns (%)",
                            labels={"x": "Month", "y": "Year", "color": "Return"}
                        )
                        fig_monthly.update_layout(height=500)
                        st.plotly_chart(fig_monthly, use_container_width=True)
                except Exception as e:
                    st.info(f"Monthly returns heatmap not available: {e}")
            else:
                st.warning("Benchmark data not available. Displaying portfolio metrics only.")
                
                # Basic metrics without benchmark
                st.metric("Portfolio Volatility", f"{equal_weight_returns.std() * np.sqrt(252):.2%}")
                st.metric("Portfolio Sharpe", f"{equal_weight_returns.mean() / equal_weight_returns.std() * np.sqrt(252):.3f}")
        else:
            st.warning("⚠️ Insufficient data for risk analysis")
    
    # ==================== TAB 5: SUPERTREND STRATEGY ====================
    if supertrend_params['enabled'] and len(tabs) > 4:
        with tabs[4]:
            st.header("🟢 Supertrend Trading Strategy")
            st.markdown("""
            Supertrend indikatörü, trend takibi için geliştirilmiş bir teknik analiz aracıdır.
            
            - **🟢 Buy Signal**: Trend aşağı yönlüden yukarı yönlüye döndüğünde
            - **🔴 Sell Signal**: Trend yukarı yönlüden aşağı yönlüye döndüğünde
            - **📊 Strategy Performance**: Backtest sonuçları ve performans metrikleri
            """)
            
            from modules.supertrend_signals import SupertrendAnalyzer, scan_multiple_stocks
            
            if supertrend_params['scan_all'] and all_ohlc_data:
                # Multi-stock scanner
                st.subheader("📊 Multi-Stock Supertrend Scanner")
                
                all_tickers_list = list(all_ohlc_data.keys())
                
                with st.spinner(f"Scanning {len(all_tickers_list)} stocks..."):
                    scan_results = scan_multiple_stocks(
                        all_tickers_list,
                        all_ohlc_data,
                        period=supertrend_params['period'],
                        multiplier=supertrend_params['multiplier']
                    )
                
                if not scan_results.empty:
                    # Display results with styling
                    st.dataframe(
                        scan_results,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        buy_signals = len(scan_results[scan_results['Action'].str.contains('BUY', na=False)])
                        st.metric("🚀 Buy Signals", buy_signals)
                    with col2:
                        sell_signals = len(scan_results[scan_results['Action'].str.contains('SELL', na=False)])
                        st.metric("📉 Sell Signals", sell_signals)
                    with col3:
                        hold_signals = len(scan_results[~scan_results['Action'].str.contains('BUY|SELL', na=False)])
                        st.metric("⏸️ Hold/Avoid", hold_signals)
                    
                    # Download button
                    csv = scan_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scanner Results",
                        data=csv,
                        file_name=f"supertrend_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime='text/csv'
                    )
                else:
                    st.info("No scan results available. Please check your data.")
            
            # Individual stock analysis
            st.subheader("🔍 Individual Stock Analysis")
            
            # Select stock for detailed analysis
            all_tickers_detailed = list(all_ohlc_data.keys())
            
            if all_tickers_detailed:
                selected_stock_supertrend = st.selectbox(
                    "Select Stock for Supertrend Analysis",
                    all_tickers_detailed,
                    key="supertrend_selector"
                )
                
                if selected_stock_supertrend and selected_stock_supertrend in all_ohlc_data:
                    ohlc_data = all_ohlc_data[selected_stock_supertrend]
                    
                    if not ohlc_data.empty and len(ohlc_data) > supertrend_params['period']:
                        # Analyze with Supertrend
                        analyzer = SupertrendAnalyzer(
                            period=supertrend_params['period'],
                            multiplier=supertrend_params['multiplier']
                        )
                        
                        signals = analyzer.generate_signals(ohlc_data)
                        
                        # Display current status
                        current_trend = signals['Trend'].iloc[-1]
                        current_price = signals['Close'].iloc[-1]
                        current_supertrend = signals['Supertrend'].iloc[-1]
                        
                        # Current status cards
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            trend_text = "🟢 UPTREND" if current_trend == 1 else "🔴 DOWNTREND"
                            st.metric("Current Trend", trend_text)
                        
                        with col2:
                            st.metric("Current Price", f"{current_price:.2f}")
                        
                        with col3:
                            if current_trend == 1:
                                distance = ((current_price - current_supertrend) / current_supertrend * 100)
                                st.metric("Distance to Supertrend", f"{distance:.2f}% (Above)")
                            else:
                                distance = ((current_supertrend - current_price) / current_supertrend * 100)
                                st.metric("Distance to Supertrend", f"{distance:.2f}% (Below)")
                        
                        with col4:
                            last_signals = signals[signals['Signal'] != 0]
                            if not last_signals.empty:
                                last_signal = last_signals.iloc[-1]
                                signal_type = "BUY" if last_signal['Signal'] == 1 else "SELL"
                                signal_color = "🟢" if last_signal['Signal'] == 1 else "🔴"
                                st.metric("Last Signal", f"{signal_color} {signal_type}")
                            else:
                                st.metric("Last Signal", "No Signal")
                        
                        # Interactive chart
                        st.subheader("📈 Supertrend Strategy Chart")
                        fig = analyzer.create_signal_chart(f"{selected_stock_supertrend} - Supertrend Analysis")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance dashboard
                        analyzer.create_performance_dashboard()
                        
                        # Strategy explanation
                        with st.expander("ℹ️ Supertrend Strategy Explained"):
                            st.markdown(f"""
                            **Current Strategy Parameters:**
                            - ATR Period: {supertrend_params['period']}
                            - ATR Multiplier: {supertrend_params['multiplier']}
                            
                            **How Supertrend Works:**
                            
                            1. **ATR (Average True Range)** hesaplanır - piyasa volatilitesini ölçer
                            2. **Basic Upper/Lower Bands** oluşturulur: (High+Low)/2 ± (multiplier × ATR)
                            3. **Final Bands** önceki değerlerle karşılaştırılarak belirlenir
                            4. **Trend Direction**: 
                               - Fiyat Final Upper Band'ı kırarsa → DOWNTREND
                               - Fiyat Final Lower Band'ı kırarsa → UPTREND
                            
                            **Signal Generation:**
                            - **BUY**: Trend DOWNTREND'den UPTREND'e döndüğünde
                            - **SELL**: Trend UPTREND'den DOWNTREND'e döndüğünde
                            
                            **Best Practices:**
                            - Düşük multiplier (2.0-2.5): Daha fazla sinyal, daha duyarlı
                            - Yüksek multiplier (3.0-4.0): Daha az sinyal, daha güvenilir
                            - Günlük grafiklerde period 7-10 arası idealdir
                            - Stop-loss için Supertrend seviyesi kullanılabilir
                            """)
                    else:
                        st.warning(f"Insufficient data for {selected_stock_supertrend}. Need at least {supertrend_params['period']} days.")
            else:
                st.info("No stock data available for Supertrend analysis.")

else:
    # Welcome screen
    st.info("👈 Please configure your market and stock selections in the sidebar, then click 'Run Analysis'")
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌍 Global Coverage")
        st.markdown("""
        - 🇹🇷 Turkish BIST 100
        - 🇺🇸 US S&P 500, NASDAQ
        - 🇪🇺 European Markets (DAX, CAC, FTSE MIB, SMI)
        - 🇯🇵 Asian Markets (Nikkei)
        - 🇦🇺 Australian ASX
        """)
    
    with col2:
        st.subheader("📊 Advanced Analytics")
        st.markdown("""
        - 🎯 PyPortfolioOpt Optimization
        - ⚠️ QuantStats Risk Metrics
        - 📈 TA-Lib Technical Indicators
        - 🕯️ Interactive OHLC Charts
        - 🟢 Supertrend Trading Strategy
        """)
    
    with col3:
        st.subheader("🎯 Professional Features")
        st.markdown("""
        - 🔄 Multi-Market Correlation
        - 📉 Efficient Frontier
        - 📊 Rolling Risk Metrics
        - 💰 Drawdown Analysis
        - 📥 Export Reports
        """)
    
    # Sample chart placeholder
    st.markdown("---")
    st.subheader("📈 Platform Preview")
    st.markdown("""
    *Select markets and stocks from the sidebar to see interactive charts and analysis.*
    
    **Supported Features:**
    - Real-time data from Yahoo Finance
    - 25+ technical indicators via TA-Lib
    - Portfolio optimization with PyPortfolioOpt
    - Professional risk metrics with QuantStats
    - Supertrend buy/sell signal generation
    """)
