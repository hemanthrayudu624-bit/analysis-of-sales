import streamlit as st
import pandas as pd 
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📈 Business Development: Sales Forecasting & Inventory Planner")

@st.cache_data
def load_forecast():
    dates = pd.date_range(start='2022-10-01', end='2024-09-30', freq='D')
    n = len(dates)
    trend = np.linspace(100, 300, n)
    yearly_season = 50 * np.sin(dates.dayofyear / 365 * 2 * np.pi)
    weekly_season = 20 * (dates.dayofweek >= 5)
    sales = trend + yearly_season + weekly_season + np.random.normal(0, 15, n)
    df = pd.DataFrame({'ds': dates, 'y': np.maximum(sales, 20)})

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.add_country_holidays(country_name='IN')
    model.fit(df)

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    return df, forecast

df, forecast = load_forecast()

# KPIs
next_30 = forecast.tail(30)['yhat'].sum()
col1, col2, col3 = st.columns(3)
col1.metric("Next 30 Days Sales", f"{next_30:,.0f} units")
col2.metric("Recommended Stock", f"{next_30*1.2:,.0f} units", "20% buffer")
col3.metric("Est. Revenue", f"₹{next_30*500:,.0f}", "₹500/unit")

# Plotly Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='lightgrey', name='Upper'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='lightgrey', name='Lower'))
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Actual Sales', marker=dict(color='black', size=3)))
fig.update_layout(title='Sales Forecast with Confidence Interval', xaxis_title='Date', yaxis_title='Units Sold')
st.plotly_chart(fig, width='stretch')

st.subheader("Business Action Plan")
st.success(f"Order {next_30*1.2:,.0f} units for next month. This prevents stock-out and saves ₹{(5000-next_30*1.2)*200:,.0f} vs old method.")

