import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta

# Set page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("📈 Stock Price Prediction App")
st.markdown("Predict stock prices for **Infosys, TCS, and Reliance**")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Stock selection
selected_stock = st.sidebar.selectbox(
    "Select Stock:",
    ["Infosys", "TCS", "Reliance"]
)

# Prediction period
prediction_days = st.sidebar.slider(
    "Prediction Period (days):",
    min_value=7,
    max_value=90,
    value=30
)

# Model parameters
st.sidebar.subheader("Model Parameters")
confidence_level = st.sidebar.slider(
    "Confidence Level:",
    min_value=0.7,
    max_value=0.95,
    value=0.85,
    step=0.05
)

# Generate sample historical data based on the selected stock
def generate_sample_data(stock_name, days=365):
    """Generate realistic sample stock data"""
    end_date = datetime.datetime.now()
    start_date = end_date - timedelta(days=days)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base prices for different stocks
    base_prices = {
        "Infosys": 1500,
        "TCS": 3200,
        "Reliance": 2400
    }
    
    base_price = base_prices.get(stock_name, 1000)
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.001, 0.02, len(dates))
    
    prices = [base_price]
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    return df

# Generate predictions using a simple model
def generate_predictions(historical_data, days=30, confidence=0.85):
    """Generate future price predictions with confidence intervals"""
    last_price = historical_data['Close'].iloc[-1]
    
    # Simple prediction model (in real app, you'd use ML models)
    daily_return_mean = historical_data['Close'].pct_change().mean()
    daily_return_std = historical_data['Close'].pct_change().std()
    
    future_dates = []
    predictions = []
    lower_bounds = []
    upper_bounds = []
    
    current_price = last_price
    z_score = {0.7: 1.04, 0.75: 1.15, 0.8: 1.28, 0.85: 1.44, 0.9: 1.645, 0.95: 1.96}
    z = z_score.get(confidence, 1.44)
    
    for i in range(1, days + 1):
        future_date = historical_data['Date'].iloc[-1] + timedelta(days=i)
        future_dates.append(future_date)
        
        # Predict next price
        predicted_return = np.random.normal(daily_return_mean, daily_return_std)
        predicted_price = current_price * (1 + predicted_return)
        predictions.append(predicted_price)
        
        # Confidence interval
        margin_error = z * daily_return_std * current_price
        lower_bounds.append(predicted_price - margin_error)
        upper_bounds.append(predicted_price + margin_error)
        
        current_price = predicted_price
    
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions,
        'Lower_Bound': lower_bounds,
        'Upper_Bound': upper_bounds
    })
    
    return prediction_df

# Main app logic
def main():
    # Generate historical data
    historical_data = generate_sample_data(selected_stock)
    
    # Generate predictions
    predictions = generate_predictions(historical_data, prediction_days, confidence_level)
    
    # Display current stock info
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = historical_data['Close'].iloc[-1]
    prev_price = historical_data['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric(
            label=f"Current Price ({selected_stock})",
            value=f"₹{current_price:.2f}",
            delta=f"{price_change_pct:.2f}%"
        )
    
    with col2:
        predicted_price = predictions['Predicted_Close'].iloc[-1]
        pred_change = ((predicted_price - current_price) / current_price) * 100
        st.metric(
            label=f"Predicted Price ({prediction_days} days)",
            value=f"₹{predicted_price:.2f}",
            delta=f"{pred_change:.2f}%"
        )
    
    with col3:
        # Calculate daily price change statistics
        daily_changes = historical_data['Close'].pct_change().dropna()
        avg_daily_change = daily_changes.mean() * 100
        st.metric(
            label="Average Daily Change",
            value=f"{avg_daily_change:.2f}%"
        )
    
    with col4:
        days_historical = len(historical_data)
        st.metric(
            label="Historical Data Points",
            value=f"{days_historical} days"
        )
    
    # Create interactive plot
    st.subheader(f"{selected_stock} Stock Price - Historical and Predicted")
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=['Stock Price Movement']
    )
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['Date'],
            y=historical_data['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Predictions
    fig.add_trace(
        go.Scatter(
            x=predictions['Date'],
            y=predictions['Predicted_Close'],
            mode='lines',
            name='Predicted',
            line=dict(color='green', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=predictions['Date'].tolist() + predictions['Date'].tolist()[::-1],
            y=predictions['Upper_Bound'].tolist() + predictions['Lower_Bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{int(confidence_level*100)}% Confidence Interval'
        ),
        row=1, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Stock Price (₹)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display data tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Historical Data")
        st.dataframe(
            historical_data.tail(10).style.format({
                'Close': '₹{:.2f}'
            }),
            use_container_width=True
        )
    
    with col2:
        st.subheader("Price Predictions")
        st.dataframe(
            predictions.head(10).style.format({
                'Predicted_Close': '₹{:.2f}',
                'Lower_Bound': '₹{:.2f}',
                'Upper_Bound': '₹{:.2f}'
            }),
            use_container_width=True
        )
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        csv_historical = historical_data.to_csv(index=False)
        st.download_button(
            label="Download Historical Data",
            data=csv_historical,
            file_name=f"{selected_stock}_historical_data.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_predictions = predictions.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv_predictions,
            file_name=f"{selected_stock}_predictions.csv",
            mime="text/csv"
        )
    
    # Additional insights
    st.subheader("📊 Trading Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        avg_return = historical_data['Close'].pct_change().mean() * 100
        st.metric("Average Daily Return", f"{avg_return:.3f}%")
    
    with insight_col2:
        total_return = ((historical_data['Close'].iloc[-1] - historical_data['Close'].iloc[0]) / historical_data['Close'].iloc[0]) * 100
        st.metric("Total Return Period", f"{total_return:.2f}%")
    
    with insight_col3:
        volatility = historical_data['Close'].pct_change().std() * 100
        st.metric("Daily Volatility", f"{volatility:.2f}%")

if __name__ == "__main__":
    main()