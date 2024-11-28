import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import time
import random
import logging
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sentiment(title):
    """Analyze sentiment of news title with financial-specific keywords"""
    try:
        positive_keywords = ['beat', 'growth', 'up', 'record', 'exceeded', 'positive', 'rise', 'FDA approval', 'Clinical trial results', 'Phase 1/2/3 trial', 'Product launch', 'New contract', 'Partnership announcement', 'Acquisition', 'Merger', 'Share buyback', 'Upgraded guidance', 'Revenue growth', 'Profitability milestone']
        negative_keywords = ['miss', 'down', 'fall', 'decline', 'negative', 'loss', 'dilution', 'lawsuit']
        
        # Get base sentiment
        blob = TextBlob(title)
        sentiment = blob.sentiment.polarity if hasattr(blob.sentiment, 'polarity') else 0
        
        title_lower = title.lower()
        has_positive = any(word in title_lower for word in positive_keywords)
        has_negative = any(word in title_lower for word in negative_keywords)
        
        if has_positive or sentiment > 0.05:
            return 'positive'
        elif has_negative or sentiment < -0.05:
            return 'negative'
        return 'neutral'
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return 'neutral'

@st.cache_data(ttl=3600)
def get_stock_news(symbol):
    """Fetch and analyze news for a stock with improved rate limiting"""
    try:
        time.sleep(1)  # Reduced delay to improve responsiveness
        
        stock = yf.Ticker(symbol)
        news = stock.news[:5] if hasattr(stock, 'news') else []
        
        if not news:
            return []
            
        analyzed_news = []
        for item in news:
            try:
                sentiment_label = analyze_sentiment(item.get('title', ''))
                analyzed_news.append({
                    'title': item.get('title', 'No title available'),
                    'link': item.get('link', '#'),
                    'sentiment': sentiment_label,
                    'date': datetime.fromtimestamp(item.get('providerPublishTime', time.time()))
                })
            except Exception as e:
                logger.warning(f"Error analyzing news item: {str(e)}")
                continue
        return analyzed_news
        
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return []

@st.cache_data(ttl=86400)
def get_stock_info(symbol):
    """Get stock information with improved error handling"""
    try:
        stock = yf.Ticker(symbol)
        fast_info = stock.fast_info
        info = stock.info if hasattr(stock, 'info') else {}
        
        # Try multiple sources for each field
        sector = info.get('sector') or info.get('industry') or 'N/A'
        industry = info.get('industry') or info.get('sector') or 'N/A'
        exchange = (info.get('exchange') or 
                   info.get('market') or 
                   fast_info.get('exchange', 'N/A'))
        
        market_cap = (info.get('marketCap') or 
                     fast_info.get('market_cap') or 
                     info.get('marketCapitalization', 0))
        
        pe_ratio = (info.get('trailingPE') or 
                   info.get('forwardPE') or 
                   fast_info.get('pe_ratio', 0))
        
        volume = (info.get('volume') or 
                 info.get('averageVolume') or 
                 fast_info.get('volume', 0))
        
        # Update institutional ownership calculation
        institutional_ownership = None
        for key in ['institutionalOwnershipPercentage', 'institutionPercentHeld', 'institutionPercent', 'institutionsPercentHeld']:
            if key in info and info[key] is not None:
                institutional_ownership = float(info[key])
                break
        
        # Format the institutional ownership with proper percentage
        institutional_ownership_str = f"{institutional_ownership*100:.1f}%" if institutional_ownership is not None else "N/A"
        
        # Get institutional count with fallback options
        institutional_count = (
            info.get('institutionsCount') or
            info.get('institutionalHolders') or
            info.get('numberOfInstitutionalHolders', 0)
        )
        
        return {
            'symbol': symbol,
            'sector': sector,
            'industry': industry,
            'exchange': exchange,
            'market_cap': format_number(market_cap, prefix="$"),
            'pe_ratio': format_number(pe_ratio),
            'volume': volume,  # Return raw volume for proper formatting later
            'description': info.get('longBusinessSummary', 'Company information not available'),
            'institutional_ownership': institutional_ownership_str,
            'institutional_count': format_number(institutional_count) if institutional_count else "N/A"
        }
    except Exception as e:
        logger.error(f"Error fetching stock info: {str(e)}")
        return None

@st.cache_data(ttl=86400)
def get_finviz_data(url):
    """Helper function to scrape Finviz data with improved error handling"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    attempt = 0
    while attempt < 5:
        try:
            logger.info(f"Attempting to fetch data from Finviz (attempt {attempt + 1})")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            table = soup.find('table', {'id': 'screener-views-table'})
            
            if not table:
                table = soup.find('table', class_='table-light')
            
            if table and hasattr(table, 'find_all'):
                rows = table.find_all('tr')[1:]  # Skip header row
                data = []
                
                for row in rows:
                    if hasattr(row, 'find_all'):
                        cols = row.find_all('td')
                        if len(cols) >= 11:
                            try:
                                ticker = cols[1].text.strip()
                                company = cols[2].text.strip()
                                
                                # Clean and parse price
                                price_str = cols[8].text.strip().replace('$', '').replace(',', '')
                                price = float(price_str) if price_str and price_str != '-' else 0
                                
                                # Clean and parse change percentage
                                change_str = cols[9].text.strip().replace('%', '').replace(',', '')
                                change = float(change_str) if change_str and change_str != '-' else 0
                                
                                # Clean and parse volume
                                volume_str = cols[10].text.strip().replace(',', '')
                                if 'K' in volume_str:
                                    volume = float(volume_str.replace('K', '')) * 1000
                                elif 'M' in volume_str:
                                    volume = float(volume_str.replace('M', '')) * 1000000
                                elif 'B' in volume_str:
                                    volume = float(volume_str.replace('B', '')) * 1000000000
                                else:
                                    volume = int(volume_str) if volume_str and volume_str != '-' else 0
                                
                                data.append({
                                    'Ticker': ticker,
                                    'Company': company,
                                    'Price': price,
                                    'Change %': change,
                                    'Volume': volume
                                })
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Error parsing row: {str(e)}")
                                continue
                
                if data:
                    logger.info(f"Successfully fetched and parsed {len(data)} stocks")
                    return pd.DataFrame(data)
            
            raise ValueError("No valid data found")
            
        except Exception as e:
            logger.error(f"Error fetching Finviz data: {str(e)}")
            if not exponential_backoff(attempt):
                break
            attempt += 1
    
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_finviz_gainers():
    """Get top 20 small-cap gainers from Finviz"""
    logger.info("Fetching top gainers from Finviz")
    url = "https://finviz.com/screener.ashx?v=111&f=cap_small&o=-change&r=1"
    df = get_finviz_data(url)
    return df.head(20) if not df.empty else df

@st.cache_data(ttl=86400)
def get_finviz_losers():
    """Get top 20 small-cap losers from Finviz"""
    logger.info("Fetching top losers from Finviz")
    url = "https://finviz.com/screener.ashx?v=111&f=cap_small&o=change&r=1"
    df = get_finviz_data(url)
    return df.head(20) if not df.empty else df

def exponential_backoff(attempt, max_attempts=5, base_delay=5):
    """Implement exponential backoff with jitter"""
    if attempt >= max_attempts:
        return False
    delay = min(300, base_delay * (2 ** attempt) + random.uniform(0, 0.1))
    time.sleep(delay)
    return True

@st.cache_data(ttl=86400)
def get_stock_price_history(symbol):
    """Fetch stock price history with error handling"""
    try:
        stock = yf.Ticker(symbol)
        history = stock.history(period='45d')
        
        if history.empty:
            logger.error(f"No price history found for {symbol}")
            return None
            
        # Clean and validate data
        history = history.replace([np.inf, -np.inf], np.nan)
        history = history.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure all required columns exist and are numeric
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in history.columns:
                logger.error(f"Missing required column: {col}")
                return None
            history[col] = pd.to_numeric(history[col], errors='coerce')
        
        return history
        
    except Exception as e:
        logger.error(f"Error fetching price history for {symbol}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators with improved validation"""
    try:
        if df is None or df.empty:
            return None
            
        indicators = df.copy()
        
        # Ensure date index is valid
        indicators.index = pd.to_datetime(indicators.index)
        
        # Initial data cleanup
        indicators = indicators.replace([np.inf, -np.inf], np.nan)
        indicators = indicators.fillna(method='ffill').fillna(method='bfill')
        
        # Validate required columns
        required_columns = ['Close', 'Volume', 'High', 'Low']
        if not all(col in indicators.columns for col in required_columns):
            return None
            
        # Calculate EMAs
        for period in [9, 20, 50]:
            indicators[f'EMA_{period}'] = indicators['Close'].ewm(span=period, adjust=False).mean()
        
        # Calculate RSI
        delta = indicators['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)  # Avoid division by zero
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp12 = indicators['Close'].ewm(span=12, adjust=False).mean()
        exp26 = indicators['Close'].ewm(span=26, adjust=False).mean()
        indicators['MACD'] = exp12 - exp26
        indicators['Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Volume metrics
        indicators['Volume_3D_Avg'] = indicators['Volume'].rolling(window=3).mean()
        
        # Calculate VWAP
        indicators['VWAP'] = (indicators['Close'] * indicators['Volume']).cumsum() / indicators['Volume'].cumsum()
        
        # Calculate Bollinger Bands
        sma = indicators['Close'].rolling(window=20).mean()
        std = indicators['Close'].rolling(window=20).std()
        indicators['BB_Upper'] = sma + (std * 2)
        indicators['BB_Middle'] = sma
        indicators['BB_Lower'] = sma - (std * 2)
        
        # Final cleanup
        indicators = indicators.replace([np.inf, -np.inf], np.nan)
        indicators = indicators.fillna(method='ffill').fillna(method='bfill')
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return None

def format_number(value, prefix="", suffix="", is_volume=False, use_compact=True):
    """Format numbers with proper separators and decimals"""
    try:
        if pd.isna(value) or value == 0:
            return "N/A"
        if isinstance(value, (int, float)):
            if is_volume:
                # Always show volumes in millions
                return f"{prefix}{value/1e6:,.2f}M{suffix}"
            else:
                # Use K/M/B suffixes for non-volume numbers
                if abs(value) >= 1e9:
                    return f"{prefix}{value/1e9:,.1f}B{suffix}"
                elif abs(value) >= 1e6:
                    return f"{prefix}{value/1e6:,.1f}M{suffix}"
                elif abs(value) >= 1e3:
                    return f"{prefix}{value/1e3:,.1f}K{suffix}"
                else:
                    return f"{prefix}{value:,.2f}{suffix}"
        return str(value)
    except Exception:
        return "N/A"

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_small_cap_symbols():
    """Fetch small-cap stocks using Finviz screener instead of Yahoo Finance"""
    try:
        logger.info("Starting small-cap symbol fetch from Finviz")
        url = "https://finviz.com/screener.ashx?v=111&f=cap_small&o=-change"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        symbols = []
        attempt = 0
        while attempt < 5:
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'lxml')
                table = soup.find('table', {'id': 'screener-views-table'})
                
                if not table:
                    table = soup.find('table', class_='table-light')
                
                if table:
                    rows = table.find_all('tr')[1:]  # Skip header row
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) > 1:
                            symbol = cols[1].text.strip()
                            symbols.append(symbol)
                    break
                else:
                    raise ValueError("Could not find screener table")
                    
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if not exponential_backoff(attempt):
                    break
                attempt += 1
        
        if not symbols:
            logger.warning("No symbols found, using backup small-cap list")
            # Backup list of common small-cap stocks
            symbols = ["SNDL", "MULN", "XELA", "AVYA", "BBIG", "NKLA", "EXPR"]
        
        logger.info(f"Found {len(symbols)} small-cap symbols")
        return symbols
        
    except Exception as e:
        logger.error(f"Error in get_small_cap_symbols: {str(e)}")
        return []

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_stock_data(symbols):
    """Fetch end-of-day stock data with batch processing"""
    if not symbols:
        return pd.DataFrame()
    
    all_data = []
    # Process in batches of 100 symbols
    for i in range(0, len(symbols), 100):
        batch_symbols = symbols[i:i+100]
        attempt = 0
        
        while attempt < 5:
            try:
                batch_data = yf.download(
                    ' '.join(batch_symbols),
                    period='1d',
                    interval='1d',
                    group_by='ticker',
                    auto_adjust=True,
                    progress=False,
                    timeout=30
                )
                
                if isinstance(batch_data, pd.DataFrame):
                    if len(batch_symbols) == 1:
                        batch_data = pd.DataFrame({batch_symbols[0]: batch_data})
                    
                    # Process each symbol in the batch
                    for symbol in batch_symbols:
                        if symbol in batch_data.columns:
                            symbol_data = batch_data[symbol].copy()
                            if not symbol_data.empty:
                                symbol_data['Symbol'] = symbol
                                all_data.append(symbol_data)
                
                break
            except Exception as e:
                if not exponential_backoff(attempt):
                    st.warning(f"Failed to fetch batch data after retries: {str(e)}")
                    break
                attempt += 1
        
        time.sleep(2)  # Rate limiting between batches
    
    if not all_data:
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(all_data, axis=0)
    return combined_data.reset_index()

def calculate_performance(data):
    """Calculate performance metrics with proper DataFrame handling"""
    if data.empty:
        return pd.DataFrame()
    
    try:
        # Ensure proper data types
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
        data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
        
        # Create performance DataFrame with explicit types
        performance = pd.DataFrame(columns=[
            'Current Price', 'Previous Close', 'Volume', 'Change %', 'Market Cap'
        ])
        
        # Group by symbol and calculate metrics
        latest_data = data.groupby('Symbol', as_index=False).last()
        
        performance['Symbol'] = latest_data['Symbol']
        performance.set_index('Symbol', inplace=True)
        
        performance['Current Price'] = latest_data['Close'].round(2)
        performance['Previous Close'] = latest_data['Open'].round(2)
        performance['Volume'] = latest_data['Volume'].astype('int64')
        
        # Calculate change percentage
        performance['Change %'] = (
            (performance['Current Price'] - performance['Previous Close']) / 
            performance['Previous Close']
        ).mul(100).round(2)
        
        # Get market caps in batch
        batch_size = 100
        symbols = performance.index.tolist()
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            attempt = 0
            
            while attempt < 5:
                try:
                    batch_tickers = yf.Tickers(' '.join(batch))
                    for symbol in batch:
                        try:
                            info = batch_tickers.tickers[symbol].fast_info
                            market_cap = info.market_cap if hasattr(info, 'market_cap') else 0
                            performance.loc[symbol, 'Market Cap'] = market_cap / 1e6  # in millions
                        except:
                            performance.loc[symbol, 'Market Cap'] = 0
                    break
                except Exception:
                    if not exponential_backoff(attempt):
                        st.warning(f"Failed to fetch market cap data for batch")
                        for symbol in batch:
                            performance.loc[symbol, 'Market Cap'] = 0
                        break
                    attempt += 1
            
            time.sleep(2)  # Rate limiting between batches
        
        # Sort and clean data
        performance = performance.sort_values('Change %', ascending=False)
        performance = performance.fillna(0)
        
        # Ensure all numeric columns have proper types
        performance['Current Price'] = performance['Current Price'].astype('float64')
        performance['Previous Close'] = performance['Previous Close'].astype('float64')
        performance['Volume'] = performance['Volume'].astype('int64')
        performance['Change %'] = performance['Change %'].astype('float64')
        performance['Market Cap'] = performance['Market Cap'].astype('float64')
        
        return performance
        
    except Exception as e:
        st.error(f"Error in performance calculation: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_stock_info(symbol):
    try:
        # Check if stock exists first
        stock = yf.Ticker(symbol)
        hist = stock.history(period='1d')
        if hist.empty:
            logger.error(f"Stock {symbol} appears to be delisted or invalid")
            return {'error': 'Stock appears to be delisted or invalid'}
        
        # Continue with existing info fetch
        info = stock.fast_info
        if not info:
            return {'error': 'Unable to fetch stock information'}
            
        return {
            'symbol': symbol,
            'sector': getattr(info, 'sector', 'N/A'),
            'industry': getattr(info, 'industry', 'N/A'),
            'exchange': getattr(info, 'exchange', 'N/A'),
            'market_cap': format_number(getattr(info, 'market_cap', 0), prefix="$"),
            'pe_ratio': format_number(getattr(info, 'pe_ratio', 0)),
            'volume': format_number(getattr(info, 'volume', 0), is_volume=True),
            'description': getattr(info, 'description', 'Company information not available'),
        }
        
    except Exception as e:
        logger.error(f"Error fetching stock info: {str(e)}")
        return {'error': f"Failed to fetch stock information: {str(e)}"}

@st.cache_data(ttl=3600)
def get_stock_price_history(symbol, period='1mo'):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            return None
        return hist[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logger.error(f"Error fetching price history for {symbol}: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for a stock"""
    try:
        # Initial data cleanup
        indicators = df.copy()
        indicators = indicators.replace([np.inf, -np.inf], np.nan)
        
        # Ensure date index is valid
        indicators.index = pd.to_datetime(indicators.index)
        
        # Basic validation
        if indicators.empty or indicators['Close'].isnull().all():
            return None
        
        # Fill missing values appropriately
        indicators['Close'] = indicators['Close'].fillna(method='ffill')
        indicators['Volume'] = indicators['Volume'].fillna(0)
        
        # Calculate indicators with validation
        # EMAs
        for period in [9, 20, 50, 200]:
            indicators[f'EMA_{period}'] = indicators['Close'].ewm(span=period, adjust=False).mean()
        
        # RSI with validation
        delta = indicators['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = indicators['Close'].ewm(span=12, adjust=False).mean()
        exp2 = indicators['Close'].ewm(span=26, adjust=False).mean()
        indicators['MACD'] = exp1 - exp2
        indicators['Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
        
        # VWAP with validation
        cumvol = indicators['Volume'].cumsum() + 1e-10
        indicators['VWAP'] = (indicators['Close'] * indicators['Volume']).cumsum() / cumvol
        
        # Bollinger Bands
        sma20 = indicators['Close'].rolling(window=20).mean()
        std20 = indicators['Close'].rolling(window=20).std()
        indicators['BB_Upper'] = sma20 + (std20 * 2)
        indicators['BB_Lower'] = sma20 - (std20 * 2)
        
        # Volume metrics
        indicators['Volume_3D_Avg'] = indicators['Volume'].rolling(window=3).mean()
        
        # Final cleanup
        indicators = indicators.replace([np.inf, -np.inf], np.nan)
        indicators = indicators.fillna(method='ffill').fillna(method='bfill')
        
        # Additional validation for chart data
        for column in ['Close', 'RSI', 'MACD', 'Signal', 'VWAP', 'BB_Upper', 'BB_Lower']:
            indicators[column] = pd.to_numeric(indicators[column], errors='coerce')
            
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return None

@st.cache_data(ttl=86400)
def get_sector_news(sector):
    """Get news related to a specific sector"""
    try:
        # Use yfinance to get sector-related news
        sector_keywords = [sector, sector.lower(), sector.replace(' ', '-')]
        all_news = []
        
        for keyword in sector_keywords:
            try:
                # Use an ETF or index as a proxy for sector news
                sector_proxy = yf.Ticker(f"^{keyword}")
                news = sector_proxy.news[:3] if hasattr(sector_proxy, 'news') else []
                
                for item in news:
                    sentiment_label = analyze_sentiment(item.get('title', ''))
                    all_news.append({
                        'title': item.get('title', 'No title available'),
                        'link': item.get('link', '#'),
                        'sentiment': sentiment_label,
                        'date': datetime.fromtimestamp(item.get('providerPublishTime', time.time()))
                    })
            except Exception:
                continue
                
        return all_news[:5]  # Return top 5 news items
        
    except Exception as e:
        logger.error(f"Error fetching sector news: {str(e)}")
        return []

@st.cache_data(ttl=86400)
def get_industry_news(industry):
    """Get news related to a specific industry"""
    try:
        # Similar approach to sector news but with industry-specific keywords
        industry_keywords = [industry, industry.lower(), industry.replace(' ', '-')]
        all_news = []
        
        for keyword in industry_keywords:
            try:
                industry_proxy = yf.Ticker(f"^{keyword}")
                news = industry_proxy.news[:3] if hasattr(industry_proxy, 'news') else []
                
                for item in news:
                    sentiment_label = analyze_sentiment(item.get('title', ''))
                    all_news.append({
                        'title': item.get('title', 'No title available'),
                        'link': item.get('link', '#'),
                        'sentiment': sentiment_label,
                        'date': datetime.fromtimestamp(item.get('providerPublishTime', time.time()))
                    })
            except Exception:
                continue
                
        return all_news[:5]  # Return top 5 news items
        
    except Exception as e:
        logger.error(f"Error fetching industry news: {str(e)}")
        return []

@st.cache_data(ttl=86400)
def get_stock_info(symbol):
    """Get stock information with improved error handling"""
    try:
        stock = yf.Ticker(symbol)
        fast_info = stock.fast_info
        info = stock.info if hasattr(stock, 'info') else {}
        
        # Get institutional data
        institutional_holders = stock.institutional_holders
        institutional_count = len(institutional_holders) if institutional_holders is not None else 0
        
        # Try multiple sources for each field
        sector = info.get('sector') or info.get('industry') or 'N/A'
        industry = info.get('industry') or info.get('sector') or 'N/A'
        exchange = (info.get('exchange') or 
                   info.get('market') or 
                   fast_info.get('exchange', 'N/A'))
        
        market_cap = (info.get('marketCap') or 
                     fast_info.get('market_cap') or 
                     info.get('marketCapitalization', 0))
        
        pe_ratio = (info.get('trailingPE') or 
                   info.get('forwardPE') or 
                   fast_info.get('pe_ratio', 0))
        
        volume = (info.get('volume') or 
                 info.get('averageVolume') or 
                 fast_info.get('volume', 0))
        
        return {
            'symbol': symbol,
            'sector': sector,
            'industry': industry,
            'exchange': exchange,
            'market_cap': format_number(market_cap, prefix="$"),
            'pe_ratio': format_number(pe_ratio),
            'volume': format_number(volume, is_volume=True),
            'description': info.get('longBusinessSummary', 'Company information not available'),
            'institutional_count': institutional_count,
            'institutional_ownership': format_number(info.get('institutionOwnership', 0), suffix="%"),
        }
    except Exception as e:
        logger.error(f"Error fetching stock info: {str(e)}")
        return None