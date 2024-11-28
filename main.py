import os
import sys
import streamlit as st
import logging

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add error handling for deployment
try:
    st.set_page_config(
        page_title="Small-Cap Stock Screener",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except Exception as e:
    st.error(f"Error initializing application: {str(e)}")
    logger.error(f"Initialization error: {str(e)}")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import time
import os
import logging
import yfinance as yf
from stock_utils import (
    get_finviz_gainers, get_finviz_losers, format_number,
    get_stock_info, get_stock_price_history, get_stock_news,
    calculate_technical_indicators
)

# Initialize session state
if 'view' not in st.session_state:
    st.session_state.view = "gainers"
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = None
if 'show_detail' not in st.session_state:
    st.session_state.show_detail = False
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = None
if 'previous_view' not in st.session_state:
    st.session_state.previous_view = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for retry logic
MAX_RETRIES = 3
RETRY_DELAY = 2

def safe_data_fetch(fetch_func, *args, **kwargs):
    """Generic retry mechanism for data fetching"""
    for attempt in range(MAX_RETRIES):
        try:
            return fetch_func(*args, **kwargs)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                st.error(f"Connection error: {str(e)}. Please refresh the page.")
                logger.error(f"Failed after {MAX_RETRIES} attempts: {str(e)}")
                return None
            time.sleep(RETRY_DELAY)
            continue

def check_connection():
    """Test connection to data service"""
    try:
        stock = yf.Ticker("AAPL")
        _ = stock.fast_info
        return True
    except Exception:
        return False

def reset_navigation_state():
    """Reset navigation state"""
    st.session_state.selected_stock = None
    st.session_state.show_detail = False
    st.session_state.chart_data = None
    st.session_state.previous_view = None

def safe_rerun():
    """Helper function to handle rerun with improved disconnection handling"""
    try:
        st.rerun()
    except Exception as e:
        st.error("Connection lost. Please refresh your browser.")
        logger.error(f"Rerun error: {str(e)}")
        time.sleep(2)
        st.rerun()

def display_navigation():
    """Display navigation buttons at the top of the page"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Top Gainers", 
                    key="gainers_btn",
                    type="primary" if st.session_state.view == "gainers" else "secondary",
                    use_container_width=True):
            st.session_state.view = "gainers"
            reset_navigation_state()
            st.rerun()
    
    with col2:
        if st.button("Top Losers",
                    key="losers_btn",
                    type="primary" if st.session_state.view == "losers" else "secondary",
                    use_container_width=True):
            st.session_state.view = "losers"
            reset_navigation_state()
            st.rerun()
    
    st.markdown("---")

def display_stock_details(symbol):
    """Display detailed view for a selected stock with improved error handling"""
    # Back button at the top with proper return to previous view
    if st.button('← Back to Screener'):
        reset_navigation_state()
        st.rerun()
        return

    # Initial stock validation
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if not info:
            st.error(f"Unable to fetch data for {symbol}. The stock might be delisted or invalid.")
            return
    except Exception as e:
        st.error(f"Error accessing stock data: {str(e)}")
        return

    with st.spinner('Loading stock details...'):
        stock_info = safe_data_fetch(get_stock_info, symbol)
        if stock_info is None:
            st.error("Failed to fetch stock information")
            return

        # Display company information
        st.title(f'📊 {symbol} Stock Details')
        
        # Update metrics display with institutional data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sector", stock_info['sector'])
            st.metric("Market Cap", stock_info['market_cap'])
        with col2:
            st.metric("Industry", stock_info['industry'])
            st.metric("PE Ratio", stock_info['pe_ratio'])
        with col3:
            st.metric("Exchange", stock_info['exchange'])
            st.metric("Volume", format_number(stock_info['volume'], is_volume=True, use_compact=True))
        with col4:
            st.metric("Institutional Investors", stock_info['institutional_count'])
            st.metric("Institutional Ownership", stock_info['institutional_ownership'])

        # Fetch price history with validation
        price_history = safe_data_fetch(get_stock_price_history, symbol)
        if price_history is not None and not price_history.empty:
            if all(col in price_history.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                indicators = safe_data_fetch(calculate_technical_indicators, price_history)
                if indicators is not None:
                    display_technical_analysis(indicators, price_history, stock_info)
                else:
                    st.warning("Unable to calculate technical indicators. Some data may be missing.")
            else:
                st.warning("Incomplete price data available. Some features may be limited.")
        else:
            st.warning("Unable to fetch price history. Some features may be limited.")

        # Display news section with sector and industry context
        st.subheader("Recent News & Analysis")
        news = get_stock_news(symbol)
        if news:
            for item in news:
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"[{item['title']}]({item['link']})")
                    with col2:
                        sentiment_color = {
                            'positive': '🟢',
                            'negative': '🔴',
                            'neutral': '🟡'
                        }.get(item['sentiment'], '⚪')
                        st.text(f"{sentiment_color} {item['sentiment'].title()}")
                    st.text(f"Date: {item['date'].strftime('%Y-%m-%d %H:%M')}")
                st.markdown("---")
        else:
            st.info("No recent news available for this stock.")

def display_stock_data(df, title):
    """Display stock data with interactive rows"""
    if df.empty:
        st.error(f"Unable to fetch {title.lower()} data. Please try again later.")
        return
    
    st.markdown(f"### {title}")
    
    # Create header row with custom styling
    header_cols = st.columns([2, 3, 2, 2, 2, 2, 1])
    header_cols[0].markdown("**Symbol**")
    header_cols[1].markdown("**Company**")
    header_cols[2].markdown("**Price**")
    header_cols[3].markdown("**Change %**")
    header_cols[4].markdown("**Volume**")
    header_cols[5].markdown("**Sentiment**")
    header_cols[6].markdown("**Action**")
    
    st.markdown("---")
    
    # Create rows for each stock with loading states
    for _, row in df.iterrows():
        cols = st.columns([2, 3, 2, 2, 2, 2, 1])
        
        # Make the entire row clickable by wrapping it in a container
        with st.container():
            cols[0].text(row['Ticker'])
            cols[1].text(row['Company'])
            cols[2].text(f"${row['Price']:.2f}")
            
            # Format change percentage with color
            change = float(row['Change %'])
            change_text = f"+{change:.2f}%" if change > 0 else f"{change:.2f}%"
            cols[3].markdown(f"<span style='color: {'green' if change > 0 else 'red'}'>{change_text}</span>", unsafe_allow_html=True)
            
            # Format volume with K/M/B notation
            cols[4].text(format_number(row['Volume'], is_volume=True, use_compact=True))
            
            # Get and display sentiment
            news = get_stock_news(row['Ticker'])
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for item in news:
                sentiment_counts[item['sentiment']] += 1
            
            # Calculate overall sentiment
            total = sum(sentiment_counts.values())
            if total > 0:
                if sentiment_counts['positive'] > sentiment_counts['negative']:
                    sentiment = '🟢 Positive'
                elif sentiment_counts['negative'] > sentiment_counts['positive']:
                    sentiment = '🔴 Negative'
                else:
                    sentiment = '🟡 Neutral'
            else:
                sentiment = '⚪ No Data'
            
            cols[5].text(sentiment)
            
            # Add View button with proper state management
            if cols[6].button('View', key=f"view_{row['Ticker']}_{title}"):
                st.session_state.selected_stock = row['Ticker']
                st.session_state.show_detail = True
                st.session_state.previous_view = title  # Store the previous view
                st.rerun()

def display_technical_analysis(indicators, price_history, info):
    """Display technical analysis section with proper validation"""
    st.subheader('Technical Analysis')
    
    # Get the RSI value early as it's used in multiple places
    rsi_value = indicators['RSI'].iloc[-1]
    
    tabs = st.tabs(['Price Chart', 'Indicators', 'Summary', 'EMAs', 'Volume', 'RSI', 'MACD', 'VWAP', 'Bollinger'])
    
    with tabs[0]:
        display_price_chart(indicators, price_history)
    
    with tabs[1]:
        display_technical_indicators(indicators)
    
    with tabs[2]:
        display_analysis_summary(indicators, info, rsi_value)

    with tabs[3]:
        st.subheader('Moving Averages (EMA)')
        fig_ema = go.Figure()
        fig_ema.add_trace(go.Scatter(x=indicators.index, y=indicators['Close'], name='Price'))
        for period in [9, 20, 50]:
            fig_ema.add_trace(go.Scatter(
                x=indicators.index,
                y=indicators[f'EMA_{period}'],
                name=f'EMA {period}',
                line=dict(width=1)
            ))
        fig_ema.update_layout(title='EMAs', height=400)
        st.plotly_chart(fig_ema, use_container_width=True)

    with tabs[4]:
        st.subheader('Volume Analysis')
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=indicators.index,
            y=indicators['Volume'],
            name='Volume'
        ))
        fig_vol.add_trace(go.Scatter(
            x=indicators.index,
            y=indicators['Volume_3D_Avg'],
            name='3-Day Avg',
            line=dict(color='orange')
        ))
        fig_vol.update_layout(title='Volume', height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Volume analysis
        current_vol = float(indicators['Volume'].iloc[-1])
        avg_vol = float(indicators['Volume_3D_Avg'].iloc[-1])
        vol_ratio = (current_vol / avg_vol - 1) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Volume", format_number(current_vol, is_volume=True))
        with col2:
            st.metric("3-Day Average", format_number(avg_vol, is_volume=True))
        with col3:
            st.metric("Volume vs Average", f"{vol_ratio:+.1f}%")

    with tabs[5]:
        st.subheader('Relative Strength Index (RSI)')
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=indicators.index,
            y=indicators['RSI'],
            name='RSI'
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title='RSI', height=400)
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # RSI Analysis
        st.markdown(f'''
        **RSI Analysis (Current: {rsi_value:.2f})**
        - Overbought (>70): Stock might be overvalued
        - Oversold (<30): Stock might be undervalued
        - Current Status: {"Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"}
        ''')

    with tabs[6]:
        st.subheader('MACD')
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=indicators.index,
            y=indicators['MACD'],
            name='MACD'
        ))
        fig_macd.add_trace(go.Scatter(
            x=indicators.index,
            y=indicators['Signal'],
            name='Signal'
        ))
        fig_macd.update_layout(title='MACD', height=400)
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # MACD Analysis
        macd = indicators['MACD'].iloc[-1]
        signal = indicators['Signal'].iloc[-1]
        st.markdown(f'''
        **MACD Analysis**
        - MACD: {macd:.3f}
        - Signal: {signal:.3f}
        - Trend: {"Bullish" if macd > signal else "Bearish"} (MACD {"above" if macd > signal else "below"} signal line)
        ''')

    with tabs[7]:
        st.subheader('VWAP')
        fig_vwap = go.Figure()
        fig_vwap.add_trace(go.Scatter(x=indicators.index, y=indicators['Close'], name='Price'))
        fig_vwap.add_trace(go.Scatter(x=indicators.index, y=indicators['VWAP'], name='VWAP'))
        fig_vwap.update_layout(title='Price vs VWAP', height=400)
        st.plotly_chart(fig_vwap, use_container_width=True)
        
        # VWAP analysis
        price = indicators['Close'].iloc[-1]
        vwap = indicators['VWAP'].iloc[-1]
        st.info(f"Price vs VWAP: {((price/vwap - 1) * 100):+.1f}%")

    with tabs[8]:
        st.subheader('Bollinger Bands')
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=indicators.index, y=indicators['Close'], name='Price'))
        fig_bb.add_trace(go.Scatter(x=indicators.index, y=indicators['BB_Upper'], name='Upper Band'))
        fig_bb.add_trace(go.Scatter(x=indicators.index, y=indicators['BB_Lower'], name='Lower Band'))
        fig_bb.update_layout(title='Bollinger Bands', height=400)
        st.plotly_chart(fig_bb, use_container_width=True)
        
        # BB analysis
        price = indicators['Close'].iloc[-1]
        upper = indicators['BB_Upper'].iloc[-1]
        lower = indicators['BB_Lower'].iloc[-1]
        bb_width = ((upper - lower) / indicators['BB_Middle'].iloc[-1]) * 100
        st.markdown(f'''
        **Bollinger Bands Analysis**
        - Current Price: ${price:.2f}
        - Upper Band: ${upper:.2f}
        - Lower Band: ${lower:.2f}
        - Band Width: {bb_width:.1f}%
        ''')

def display_price_chart(indicators, price_history):
    """Display the price chart with moving averages and volume analysis"""
    # Create subplots with price chart and volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03,
                       row_heights=[0.7, 0.3])

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=indicators.index,
        open=pd.to_numeric(price_history['Open'], errors='coerce'),
        high=pd.to_numeric(price_history['High'], errors='coerce'),
        low=pd.to_numeric(price_history['Low'], errors='coerce'),
        close=pd.to_numeric(price_history['Close'], errors='coerce'),
        name='Price'
    ), row=1, col=1)

    # Add EMA lines to price chart
    for period in [9, 20, 50]:
        fig.add_trace(go.Scatter(
            x=indicators.index,
            y=indicators[f'EMA_{period}'],
            name=f'EMA {period}',
            line=dict(width=1)
        ), row=1, col=1)

    # Calculate colors for volume bars
    colors = ['green' if price_history['Close'].iloc[i] >= price_history['Open'].iloc[i] else 'red' 
             for i in range(len(price_history))]

    # Add volume bars with color
    fig.add_trace(go.Bar(
        x=indicators.index,
        y=price_history['Volume'],
        name='Volume',
        marker_color=colors
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title='Price and Volume',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

def display_analysis_summary(indicators, info, rsi_value):
    """Display summary of technical analysis"""
    st.subheader('Analysis Summary')
    
    # Price trend
    current_price = indicators['Close'].iloc[-1]
    prev_price = indicators['Close'].iloc[-2]
    price_change = ((current_price / prev_price) - 1) * 100
    
    # Volume analysis
    current_vol = indicators['Volume'].iloc[-1]
    avg_vol = indicators['Volume_3D_Avg'].iloc[-1]
    vol_change = ((current_vol / avg_vol) - 1) * 100
    
    # MACD analysis
    macd = indicators['MACD'].iloc[-1]
    signal = indicators['Signal'].iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Price Change", f"{price_change:+.2f}%")
        st.metric("Volume vs Average", f"{vol_change:+.2f}%")
        
    with col2:
        st.metric("RSI", f"{rsi_value:.1f}")
        st.metric("MACD Signal", "Bullish" if macd > signal else "Bearish")

def display_technical_indicators(indicators):
    """Display technical indicators summary"""
    if indicators is None:
        st.warning("No indicator data available")
        return
        
    # Create metrics for key indicators
    col1, col2, col3 = st.columns(3)
    
    # RSI Status
    rsi = indicators['RSI'].iloc[-1]
    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    with col1:
        st.metric("RSI", f"{rsi:.2f}", rsi_status)
    
    # MACD Signal
    macd = indicators['MACD'].iloc[-1]
    signal = indicators['Signal'].iloc[-1]
    macd_trend = "Bullish" if macd > signal else "Bearish"
    with col2:
        st.metric("MACD", f"{macd:.3f}", f"{macd_trend}")
    
    # Volume Analysis
    current_vol = indicators['Volume'].iloc[-1]
    avg_vol = indicators['Volume_3D_Avg'].iloc[-1]
    vol_change = ((current_vol / avg_vol) - 1) * 100
    with col3:
        st.metric("Volume vs Avg", f"{vol_change:+.1f}%", 
                 "Above Average" if vol_change > 0 else "Below Average")

# Initialize app state and configuration
try:
    # Clear all caches at startup
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("Cache cleared successfully")

    # Page config must be the first Streamlit command
    st.set_page_config(
        page_title="Small-Cap Stock Screener",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    logger.info("Page configuration set successfully")

    # Load custom CSS
    css_path = os.path.join('assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Initialize session state
    if 'show_detail' not in st.session_state:
        st.session_state.show_detail = False
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = None
    if 'chart_data' not in st.session_state:
        st.session_state.chart_data = None
    if 'previous_view' not in st.session_state:
        st.session_state.previous_view = None
    if 'view' not in st.session_state:
        st.session_state.view = "gainers"

except Exception as e:
    logger.error(f"Error during application startup: {str(e)}")
    st.error("An error occurred during application startup. Please refresh the page.")

# Main application code
if __name__ == "__main__":
    try:
        # Display navigation
        display_navigation()
        
        # Show stock data based on view
        if st.session_state.show_detail and st.session_state.selected_stock:
            display_stock_details(st.session_state.selected_stock)
        else:
            if st.session_state.view == "gainers":
                df = safe_data_fetch(get_finviz_gainers)
                if df is not None:
                    display_stock_data(df, "Top Gainers")
            else:
                df = safe_data_fetch(get_finviz_losers)
                if df is not None:
                    display_stock_data(df, "Top Losers")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please refresh the page.")
