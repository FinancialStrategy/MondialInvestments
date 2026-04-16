"""
Portfolio Optimization Module with PyPortfolioOpt
Professional portfolio optimization and efficient frontier
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple

# PyPortfolioOpt imports
try:
    from pypfopt import expected_returns, risk_models
    from pypfopt import EfficientFrontier, objective_functions
    from pypfopt import plotting
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    print("PyPortfolioOpt not available")


class PortfolioOptimizer:
    """Portfolio optimization using PyPortfolioOpt"""
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize optimizer with price data
        
        Args:
            price_data: DataFrame with daily prices, columns = tickers
        """
        self.prices = price_data.dropna(axis=1, how='all')
        
        if len(self.prices.columns) < 2:
            st.warning("Need at least 2 assets for portfolio optimization")
            self.mu = None
            self.S = None
        else:
            # Calculate expected returns and covariance matrix
            self.mu = expected_returns.mean_historical_return(self.prices)
            self.S = risk_models.sample_cov(self.prices)
    
    def optimize_max_sharpe(self) -> Dict:
        """Optimize for maximum Sharpe ratio"""
        if self.mu is None or self.S is None:
            return self._empty_result()
        
        try:
            ef = EfficientFrontier(self.mu, self.S)
            ef.add_constraint(lambda w: w >= 0)  # No short selling
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False)
            
            return {
                'weights': cleaned_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'status': 'success'
            }
        except Exception as e:
            st.error(f"Max Sharpe optimization failed: {e}")
            return self._empty_result()
    
    def optimize_min_volatility(self) -> Dict:
        """Optimize for minimum volatility"""
        if self.mu is None or self.S is None:
            return self._empty_result()
        
        try:
            ef = EfficientFrontier(self.mu, self.S)
            ef.add_constraint(lambda w: w >= 0)
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False)
            
            return {
                'weights': cleaned_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'status': 'success'
            }
        except Exception as e:
            st.error(f"Min volatility optimization failed: {e}")
            return self._empty_result()
    
    def optimize_max_quadratic_utility(self, risk_aversion: float = 3.0) -> Dict:
        """Optimize for maximum quadratic utility"""
        if self.mu is None or self.S is None:
            return self._empty_result()
        
        try:
            ef = EfficientFrontier(self.mu, self.S)
            ef.add_constraint(lambda w: w >= 0)
            weights = ef.max_quadratic_utility(risk_aversion)
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False)
            
            return {
                'weights': cleaned_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'status': 'success'
            }
        except Exception as e:
            st.error(f"Quadratic utility optimization failed: {e}")
            return self._empty_result()
    
    def optimize_efficient_return(self, target_return: float) -> Dict:
        """Optimize for a target return"""
        if self.mu is None or self.S is None:
            return self._empty_result()
        
        try:
            ef = EfficientFrontier(self.mu, self.S)
            ef.add_constraint(lambda w: w >= 0)
            weights = ef.efficient_return(target_return)
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False)
            
            return {
                'weights': cleaned_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'status': 'success'
            }
        except Exception as e:
            st.error(f"Efficient return optimization failed: {e}")
            return self._empty_result()
    
    def optimize_efficient_risk(self, target_volatility: float) -> Dict:
        """Optimize for a target risk level"""
        if self.mu is None or self.S is None:
            return self._empty_result()
        
        try:
            ef = EfficientFrontier(self.mu, self.S)
            ef.add_constraint(lambda w: w >= 0)
            weights = ef.efficient_risk(target_volatility)
            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False)
            
            return {
                'weights': cleaned_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'status': 'success'
            }
        except Exception as e:
            st.error(f"Efficient risk optimization failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict:
        """Return empty result when optimization fails"""
        return {
            'weights': {},
            'expected_return': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'status': 'failed'
        }
    
    def get_efficient_frontier_points(self, points: int = 50) -> pd.DataFrame:
        """Generate efficient frontier points"""
        if self.mu is None or self.S is None:
            return pd.DataFrame()
        
        try:
            frontiers = []
            min_return = self.mu.min()
            max_return = self.mu.max()
            
            for target_return in np.linspace(min_return, max_return, points):
                try:
                    ef = EfficientFrontier(self.mu, self.S)
                    ef.add_constraint(lambda w: w >= 0)
                    weights = ef.efficient_return(target_return)
                    ret, vol, _ = ef.portfolio_performance(verbose=False)
                    frontiers.append({
                        'return': ret,
                        'volatility': vol,
                        'sharpe': ret / vol if vol > 0 else 0
                    })
                except:
                    continue
            
            return pd.DataFrame(frontiers)
        except Exception as e:
            st.error(f"Error generating efficient frontier: {e}")
            return pd.DataFrame()
    
    def plot_efficient_frontier(self) -> go.Figure:
        """Create interactive efficient frontier plot"""
        ef_points = self.get_efficient_frontier_points()
        
        if ef_points.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Efficient frontier line
        fig.add_trace(go.Scatter(
            x=ef_points['volatility'],
            y=ef_points['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=2)
        ))
        
        # Individual assets
        if self.mu is not None:
            asset_volatilities = np.sqrt(np.diag(self.S))
            
            fig.add_trace(go.Scatter(
                x=asset_volatilities,
                y=self.mu.values,
                mode='markers',
                name='Individual Assets',
                marker=dict(size=10, color='red', symbol='circle'),
                text=self.mu.index,
                textposition='top center'
            ))
        
        fig.update_layout(
            title='Mean-Variance Efficient Frontier',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            template='plotly_white',
            height=500,
            hovermode='closest'
        )
        
        fig.update_xaxis(tickformat='.2%')
        fig.update_yaxis(tickformat='.2%')
        
        return fig
    
    def create_optimization_dashboard(self) -> Dict:
        """Create interactive optimization dashboard"""
        st.subheader("📊 Portfolio Optimization Results")
        
        if len(self.prices.columns) < 2:
            st.warning("⚠️ Please select at least 2 assets for portfolio optimization")
            return self._empty_result()
        
        col1, col2, col3 = st.columns(3)
        
        # Optimization strategy selection
        strategy = st.selectbox(
            "🎯 Optimization Strategy",
            ["Max Sharpe Ratio", "Min Volatility", "Max Quadratic Utility"]
        )
        
        if strategy == "Max Sharpe Ratio":
            result = self.optimize_max_sharpe()
        elif strategy == "Min Volatility":
            result = self.optimize_min_volatility()
        else:
            risk_aversion = st.slider("⚡ Risk Aversion Parameter", 1.0, 10.0, 3.0, 0.5)
            result = self.optimize_max_quadratic_utility(risk_aversion)
        
        if result['status'] == 'success':
            with col1:
                st.metric("📈 Expected Annual Return", f"{result['expected_return']:.2%}")
            with col2:
                st.metric("📉 Annual Volatility", f"{result['volatility']:.2%}")
            with col3:
                st.metric("🎯 Sharpe Ratio", f"{result['sharpe_ratio']:.3f}")
            
            # Weight distribution
            if result['weights']:
                st.subheader("💰 Optimal Portfolio Weights")
                
                weights_df = pd.DataFrame.from_dict(
                    result['weights'], 
                    orient='index', 
                    columns=['Weight']
                ).sort_values('Weight', ascending=False)
                
                # Filter weights > 0.01 for display
                weights_df = weights_df[weights_df['Weight'] > 0.01]
                
                fig = px.bar(
                    weights_df,
                    x=weights_df.index,
                    y='Weight',
                    title='Portfolio Allocation',
                    color='Weight',
                    color_continuous_scale='Viridis',
                    text=weights_df['Weight'].apply(lambda x: f'{x:.1%}')
                )
                fig.update_layout(
                    xaxis_title="Asset",
                    yaxis_title="Portfolio Weight",
                    height=400,
                    template='plotly_white'
                )
                fig.update_yaxis(tickformat='.0%')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Download weights
                csv = weights_df.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Portfolio Weights",
                    data=csv,
                    file_name='portfolio_weights.csv',
                    mime='text/csv'
                )
        else:
            st.error("❌ Optimization failed. Please try different parameters.")
        
        return result


def calculate_portfolio_statistics(returns: pd.DataFrame, weights: Dict) -> Dict:
    """
    Calculate portfolio statistics given weights
    
    Args:
        returns: Daily returns DataFrame
        weights: Dictionary of weights {ticker: weight}
    
    Returns:
        Dictionary of portfolio statistics
    """
    if not weights or returns.empty:
        return {}
    
    # Convert weights to Series
    weight_series = pd.Series(weights)
    weight_series = weight_series / weight_series.sum()  # Normalize
    
    # Align returns and weights
    common_tickers = [t for t in weight_series.index if t in returns.columns]
    if not common_tickers:
        return {}
    
    weight_series = weight_series[common_tickers]
    portfolio_returns = returns[common_tickers].dot(weight_series)
    
    # Calculate statistics
    annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
    
    # Calculate drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'expected_return': annual_return,
        'volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'weights': weight_series.to_dict()
    }
