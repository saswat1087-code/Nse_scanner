import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page for mobile
st.set_page_config(
    page_title="NSE Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for mobile responsiveness
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px;
        font-size: 16px;
    }
    .bullish {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    .bearish {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 10px 0;
    }
    .neutral {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    @media (max-width: 768px) {
        .stMarkdown {
            font-size: 14px;
        }
        h1 {
            font-size: 24px !important;
        }
        h3 {
            font-size: 18px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

class NSETechnicalAnalyzer:
    """NSE Stock Technical Analysis Engine"""
    
    def __init__(self):
        self.nifty_50_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
            'AXISBANK.NS', 'LT.NS', 'HCLTECH.NS', 'WIPRO.NS', 'MARUTI.NS',
            'SUNPHARMA.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS'
        ]
    
    def get_stock_data(self, symbol, period='3mo'):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if df.empty:
                return None
            return df
        except Exception as e:
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate various technical indicators"""
        if df is None or len(df) < 20:
            return None
        
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Support and Resistance
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        return df
    
    def detect_patterns(self, df):
        """Detect chart patterns and generate signals"""
        signals = {
            'bullish': [],
            'bearish': [],
            'neutral': []
        }
        
        if df is None or len(df) < 50:
            return signals
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Moving Average Crossover
        if prev['SMA_20'] <= prev['SMA_50'] and latest['SMA_20'] > latest['SMA_50']:
            signals['bullish'].append("Golden Cross (20 SMA above 50 SMA)")
        elif prev['SMA_20'] >= prev['SMA_50'] and latest['SMA_20'] < latest['SMA_50']:
            signals['bearish'].append("Death Cross (20 SMA below 50 SMA)")
        
        # RSI Signals
        if latest['RSI'] < 30:
            signals['bullish'].append(f"Oversold (RSI = {latest['RSI']:.1f})")
        elif latest['RSI'] > 70:
            signals['bearish'].append(f"Overbought (RSI = {latest['RSI']:.1f})")
        
        # MACD Signals
        if prev['MACD'] <= prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']:
            signals['bullish'].append("MACD Bullish Crossover")
        elif prev['MACD'] >= prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']:
            signals['bearish'].append("MACD Bearish Crossover")
        
        # Bollinger Band Signals
        if latest['Close'] <= latest['BB_Lower']:
            signals['bullish'].append("At Lower Bollinger Band (Potential Bounce)")
        elif latest['Close'] >= latest['BB_Upper']:
            signals['bearish'].append("At Upper Bollinger Band (Potential Reversal)")
        
        # Volume Confirmation
        if latest['Volume_Ratio'] > 1.5 and latest['Close'] > latest['Open']:
            signals['bullish'].append("High Volume Bullish Bar")
        elif latest['Volume_Ratio'] > 1.5 and latest['Close'] < latest['Open']:
            signals['bearish'].append("High Volume Bearish Bar")
        
        return signals
    
    def calculate_score(self, signals):
        """Calculate overall bullish/bearish score"""
        bullish_count = len(signals['bullish'])
        bearish_count = len(signals['bearish'])
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0, 'Neutral'
        
        score = (bullish_count / total) * 100
        if score >= 60:
            sentiment = 'Bullish'
        elif score <= 40:
            sentiment = 'Bearish'
        else:
            sentiment = 'Neutral'
        
        return score, sentiment
    
    def scan_stocks(self, selected_symbols=None):
        """Scan multiple stocks and generate analysis"""
        if selected_symbols is None:
            selected_symbols = self.nifty_50_symbols
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(selected_symbols):
            status_text.text(f"Scanning {symbol}... ({idx+1}/{len(selected_symbols)})")
            progress_bar.progress((idx + 1) / len(selected_symbols))
            
            df = self.get_stock_data(symbol)
            if df is None:
                continue
            
            df = self.calculate_technical_indicators(df)
            if df is None:
                continue
            
            signals = self.detect_patterns(df)
            score, sentiment = self.calculate_score(signals)
            
            current_price = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            price_change = ((current_price - prev_close) / prev_close) * 100
            
            results.append({
                'Symbol': symbol.replace('.NS', ''),
                'Price': round(current_price, 2),
                'Change %': round(price_change, 2),
                'RSI': round(df['RSI'].iloc[-1], 1),
                'Volume Ratio': round(df['Volume_Ratio'].iloc[-1], 2),
                'Sentiment': sentiment,
                'Score': round(score, 1),
                'Bullish Signals': len(signals['bullish']),
                'Bearish Signals': len(signals['bearish']),
                'Signals': signals,
                'Data': df
            })
        
        progress_bar.empty()
        status_text.empty()
        return pd.DataFrame(results)
    
    def plot_chart(self, df, symbol):
        """Create interactive chart with indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol} - Price & Indicators', 'RSI', 'MACD')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red', width=2)),
            row=3, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', marker_color=colors),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f"{symbol} - Technical Analysis Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=700,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig

def main():
    st.title("📊 NSE Stock Scanner - Bullish & Bearish Analysis")
    st.markdown("---")
    
    analyzer = NSETechnicalAnalyzer()
    
    # Sidebar - Collapsible on mobile
    with st.sidebar:
        st.header("🔍 Scanner Settings")
        
        scan_option = st.radio(
            "Select stocks to scan:",
            ["Nifty 50 Stocks", "Single Stock"],
            help="Choose which stocks to analyze"
        )
        
        if scan_option == "Single Stock":
            single_stock = st.text_input("Enter stock symbol", "RELIANCE.NS")
            selected_symbols = [single_stock]
        else:
            selected_symbols = analyzer.nifty_50_symbols
            st.info(f"Scanning {len(selected_symbols)} Nifty 50 stocks")
        
        period = st.selectbox(
            "Analysis Period",
            ["1mo", "3mo", "6mo", "1y"],
            index=1
        )
        
        st.markdown("---")
        st.subheader("📈 Filter Parameters")
        min_rsi = st.slider("Min RSI", 0, 100, 30)
        max_rsi = st.slider("Max RSI", 0, 100, 70)
        min_volume = st.slider("Min Volume Ratio", 0.5, 3.0, 1.0)
        
        scan_button = st.button("🚀 Start Scanning", use_container_width=True)
    
    if scan_button:
        with st.spinner("Scanning stocks... This may take a moment."):
            results_df = analyzer.scan_stocks(selected_symbols)
            
            if results_df.empty:
                st.warning("No data retrieved. Please check stock symbols.")
                return
            
            # Apply filters
            filtered_df = results_df[
                (results_df['RSI'] >= min_rsi) & 
                (results_df['RSI'] <= max_rsi) &
                (results_df['Volume Ratio'] >= min_volume)
            ]
            
            # Display results
            st.subheader("📋 Scan Results")
            
            # Metrics row
            col1, col2, col3 = st.columns(3)
            bullish_count = len(filtered_df[filtered_df['Sentiment'] == 'Bullish'])
            bearish_count = len(filtered_df[filtered_df['Sentiment'] == 'Bearish'])
            neutral_count = len(filtered_df[filtered_df['Sentiment'] == 'Neutral'])
            
            with col1:
                st.metric("🐂 Bullish", bullish_count)
            with col2:
                st.metric("🐻 Bearish", bearish_count)
            with col3:
                st.metric("⚖️ Neutral", neutral_count)
            
            # Display table
            display_cols = ['Symbol', 'Price', 'Change %', 'RSI', 'Sentiment', 'Score']
            st.dataframe(
                filtered_df[display_cols].style.background_gradient(
                    subset=['Score'], cmap='RdYlGn', vmin=0, vmax=100
                ),
                use_container_width=True,
                height=400
            )
            
            # Detailed view
            if not filtered_df.empty:
                st.markdown("---")
                st.subheader("🔍 Detailed Analysis")
                
                selected_stock = st.selectbox(
                    "Select a stock for detailed analysis:",
                    filtered_df['Symbol'].tolist()
                )
                
                if selected_stock:
                    stock_data = filtered_df[filtered_df['Symbol'] == selected_stock].iloc[0]
                    sentiment = stock_data['Sentiment']
                    
                    # Sentiment card
                    sentiment_class = sentiment.lower()
                    st.markdown(f"""
                    <div class='{sentiment_class}'>
                        <h3>{selected_stock} - {sentiment} Signal</h3>
                        <p><strong>Score:</strong> {stock_data['Score']}/100</p>
                        <p><strong>Price:</strong> ₹{stock_data['Price']} 
                        ({'+' if stock_data['Change %'] > 0 else ''}{stock_data['Change %']}%)</p>
                        <p><strong>RSI:</strong> {stock_data['RSI']} | 
                        <strong>Volume:</strong> {stock_data['Volume Ratio']}x avg</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Signals
                    col1, col2 = st.columns(2)
                    signals = stock_data['Signals']
                    
                    with col1:
                        st.markdown("### 🟢 Bullish Signals")
                        if signals['bullish']:
                            for signal in signals['bullish']:
                                st.success(f"✓ {signal}")
                        else:
                            st.info("No bullish signals")
                    
                    with col2:
                        st.markdown("### 🔴 Bearish Signals")
                        if signals['bearish']:
                            for signal in signals['bearish']:
                                st.error(f"✗ {signal}")
                        else:
                            st.info("No bearish signals")
                    
                    # Chart
                    st.markdown("### 📊 Technical Chart")
                    fig = analyzer.plot_chart(stock_data['Data'], selected_stock)
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👈 Configure settings in sidebar and click 'Start Scanning'")
        
        with st.expander("ℹ️ How to use this scanner"):
            st.markdown("""
            **Features:**
            - Scans Nifty 50 stocks or individual stocks
            - Analyzes RSI, MACD, Moving Averages, Bollinger Bands
            - Detects bullish and bearish patterns
            - Provides sentiment score (0-100)
            - Interactive charts with all indicators
            
            **Bullish signals include:**
            - Golden Cross
            - RSI oversold recovery
            - MACD bullish crossover
            - High volume breakout
            
            **Bearish signals include:**
            - Death Cross
            - RSI overbought
            - MACD bearish crossover
            - High volume breakdown
            """)

if __name__ == "__main__":
    main()