import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("📊 Business Development: Sales Forecasting & Inventory Planner")

# 1. File Upload - CSV or Excel
uploaded_file = st.file_uploader("Mee Sales file upload cheyandi - CSV or Excel", 
                                 type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        # 2. File type batti read chey
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"File loaded: {uploaded_file.name}")
        st.write("Data Preview:", df.head())
        
        # 3. Columns select - Auto guess kuda chestadi
        cols = df.columns.tolist()
        
        # Date column auto guess
        date_guess = None
        for col in cols:
            if 'date' in col.lower() or 'day' in col.lower():
                date_guess = col
                break
        
        # Sales column auto guess  
        sales_guess = None
        for col in cols:
            if 'sale' in col.lower() or 'revenue' in col.lower() or 'amount' in col.lower():
                sales_guess = col
                break
        
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox("Date Column select chey", cols, 
                                    index=cols.index(date_guess) if date_guess else 0)
        with col2:
            sales_col = st.selectbox("Sales Column select chey", cols, 
                                     index=cols.index(sales_guess) if sales_guess else 0)
        
        # 4. Prophet kosam prepare chey - Error proof
        df_prophet = df[[date_col, sales_col]].copy()
        df_prophet.columns = ['ds', 'y']
        
        # Date convert chey - format edaina handle avutadi
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], errors='coerce')
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        
        # Nulls teesey
        df_prophet = df_prophet.dropna()
        
        if df_prophet.empty:
            st.error("Date or Sales column lo data sarigga ledu. Check cheyandi.")
        else:
            # 5. Model train chey
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            model.add_country_holidays(country_name='IN')
            model.fit(df_prophet)
            
            # 6. Forecast chey
            periods = st.slider("Enni rojulu forecast kavali?", 7, 365, 90)
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # 7. Output chupinchu
            st.subheader("Forecast Results")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Actual'))
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("Next 10 days forecast:", forecast[['ds', 'yhat']].tail(10))
            st.download_button("Download Forecast CSV", 
                               forecast.to_csv(index=False), 
                               file_name="forecast.csv")
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("CSV/Excel lo 'Date' and 'Sales' columns unnaya check cheyandi")
        
else:
    st.info("👆 Painna CSV or Excel file upload cheyandi")
# Forecast table kindha ee code petti
st.subheader("📦 Inventory Planner")
safety_stock = st.number_input("Safety Stock %", 10, 50, 20)
lead_time = st.number_input("Supplier Lead Time - Days", 1, 30, 7)

avg_daily_sales = forecast['yhat'].tail(30).mean()
reorder_point = avg_daily_sales * lead_time * (1 + safety_stock/100)

st.metric("Daily Avg Sales", f"{avg_daily_sales:.0f} units")
st.metric("Reorder Point", f"{reorder_point:.0f} units")
st.warning(f"Stock {reorder_point:.0f} kanna takkuva ayithe malli order pettu")
