import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("Business Development: Sales Forecasting & Inventory Planner")

# 1. FILE UPLOAD - IDHI MISSING NEE APP LO
uploaded_file = st.file_uploader("Mee Sales CSV file upload cheyandi", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # User ki columns select chesey option ivvu
    col1, col2 = st.columns(2)
    with col1:
        date_col = st.selectbox("Date Column select chey", df.columns)
    with col2:
        sales_col = st.selectbox("Sales Column select chey", df.columns)
    
    # Prophet ki format chey
    df_prophet = df[[date_col, sales_col]].rename(columns={date_col: 'ds', sales_col: 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    
    # Model run chey
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Nee metrics chupinchu
    next_30_sales = forecast['yhat'][-30:].sum()
    st.metric("Next 30 Days Sales", f"{int(next_30_sales):,} units")
    
    # Graph plot chey - nee screenshot loni graph code ikkada raay
    
else:
    st.info("Paine CSV file upload chey bro. Appudu forecast vastadi.")


