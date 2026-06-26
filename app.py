import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #8892b0; font-size: 13px; font-weight: 600;
                       letter-spacing: 1.2px; text-transform: uppercase; margin: 0 0 8px; }
    .metric-card .value { color: #ccd6f6; font-size: 28px; font-weight: 700; margin: 0; }
    .metric-card .delta { font-size: 13px; margin-top: 4px; }
    .delta-pos { color: #64ffda; }
    .delta-neg { color: #ff6b6b; }

    .section-header {
        color: #ccd6f6;
        font-size: 18px;
        font-weight: 600;
        border-left: 3px solid #64ffda;
        padding-left: 12px;
        margin: 28px 0 16px;
    }

    .upload-zone {
        border: 2px dashed #2d3561;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        background: #1a1d2e;
        transition: border-color 0.2s;
    }

    .stDataFrame { border-radius: 8px; overflow: hidden; }

    div[data-testid="stMetric"] {
        background: #1a1d2e;
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label { color: #8892b0 !important; font-size: 12px !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #ccd6f6 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] svg { display: none; }

    .stSelectbox > div > div { background-color: #1a1d2e; border-color: #2d3561; }
    .stSlider > div > div { background-color: #64ffda20; }

    .insight-box {
        background: linear-gradient(135deg, #0d2137 0%, #0a1628 100%);
        border: 1px solid #1e4a7c;
        border-left: 4px solid #64ffda;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
        color: #a8b2d8;
        font-size: 14px;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 8px 0 24px;">
    <h1 style="color:#ccd6f6; font-size:32px; font-weight:700; margin:0;">
        📈 Sales Forecasting Dashboard
    </h1>
    <p style="color:#8892b0; margin:6px 0 0; font-size:15px;">
        Upload your sales data · Explore trends · Generate forecasts
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font=dict(color="#8892b0", family="Inter"),
    xaxis=dict(gridcolor="#1a1d2e", linecolor="#2d3561"),
    yaxis=dict(gridcolor="#1a1d2e", linecolor="#2d3561"),
    margin=dict(l=20, r=20, t=40, b=20),
)

TEAL   = "#64ffda"
PURPLE = "#7b68ee"
CORAL  = "#ff6b6b"
GOLD   = "#ffd700"


def load_file(uploaded):
    """Read uploaded Excel or CSV into a DataFrame."""
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    elif name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(uploaded)
    else:
        st.error("Unsupported file type. Please upload a .csv or .xlsx file.")
        return None


def detect_date_and_sales(df: pd.DataFrame):
    """Heuristically find date column and numeric sales column."""
    date_col, sales_col = None, None

    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.7:
                    df[c] = parsed
                    date_col = c
                    break
            except Exception:
                pass

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != date_col]
    sales_keywords = ["sales", "revenue", "amount", "value", "total", "price", "qty", "quantity", "units"]
    for kw in sales_keywords:
        for c in numeric_cols:
            if kw in c.lower():
                sales_col = c
                break
        if sales_col:
            break
    if not sales_col and numeric_cols:
        sales_col = numeric_cols[0]

    return date_col, sales_col


def build_forecast(series: pd.Series, periods: int, model_type: str):
    """Return forecast values and in-sample metrics."""
    y = series.dropna().values
    X = np.arange(len(y)).reshape(-1, 1)

    if model_type == "Linear Regression":
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
        forecast = reg.predict(future_X)

    elif model_type == "Polynomial (degree 2)":
        poly = PolynomialFeatures(2)
        X_p = poly.fit_transform(X)
        reg = LinearRegression().fit(X_p, y)
        y_pred = reg.predict(X_p)
        future_X = poly.transform(np.arange(len(y), len(y) + periods).reshape(-1, 1))
        forecast = reg.predict(future_X)

    elif model_type == "Moving Average":
        window = min(3, len(y))
        ma = pd.Series(y).rolling(window).mean().fillna(method="bfill").values
        y_pred = ma
        last_avg = np.mean(y[-window:])
        forecast = np.full(periods, last_avg)

    elif model_type == "Exponential Smoothing":
        alpha = 0.3
        smoothed = [y[0]]
        for v in y[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
        y_pred = np.array(smoothed)
        forecast = np.full(periods, smoothed[-1])

    mae  = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2   = r2_score(y, y_pred)
    return forecast, y_pred, mae, rmse, r2


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    uploaded = st.file_uploader(
        "Upload Excel or CSV",
        type=["csv", "xlsx", "xls", "xlsm"],
        help="Supported: .csv, .xlsx, .xls, .xlsm",
    )

    st.markdown("---")
    model_type = st.selectbox(
        "Forecast Model",
        ["Linear Regression", "Polynomial (degree 2)", "Moving Average", "Exponential Smoothing"],
    )
    forecast_periods = st.slider("Forecast Periods", 3, 24, 6)

    st.markdown("---")
    st.markdown(
        "<small style='color:#4a5568;'>Upload data above to enable column selectors.</small>",
        unsafe_allow_html=True,
    )


# ─── Demo data fallback ─────────────────────────────────────────────────────
def make_demo():
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    trend = np.linspace(12000, 28000, 36)
    seasonality = 3000 * np.sin(np.linspace(0, 4 * np.pi, 36))
    noise = np.random.normal(0, 800, 36)
    sales = trend + seasonality + noise
    region_map = ["North", "South", "East", "West"]
    product_map = ["Product A", "Product B", "Product C"]
    df = pd.DataFrame({
        "Date": dates,
        "Sales": sales.round(2),
        "Region": np.random.choice(region_map, 36),
        "Product": np.random.choice(product_map, 36),
        "Units": (sales / np.random.uniform(50, 150, 36)).astype(int),
        "Cost":  (sales * np.random.uniform(0.55, 0.70, 36)).round(2),
    })
    return df


# ─── Main ───────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👆 Upload a file in the sidebar, or explore the **demo dataset** below.")
    use_demo = st.checkbox("Use demo dataset", value=True)
    df = make_demo() if use_demo else None
else:
    df = load_file(uploaded)

if df is None:
    st.stop()

# ── Auto-detect columns ──────────────────────────────────────────────────────
auto_date, auto_sales = detect_date_and_sales(df.copy())

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Column Mapping")
    date_col  = st.selectbox("Date Column",  df.columns.tolist(),
                              index=df.columns.tolist().index(auto_date)  if auto_date  in df.columns else 0)
    sales_col = st.selectbox("Sales Column", [c for c in df.columns if c != date_col],
                              index=([c for c in df.columns if c != date_col].index(auto_sales))
                              if auto_sales and auto_sales in df.columns else 0)

    cat_cols = [c for c in df.columns if df[c].dtype == object and c not in [date_col, sales_col]]
    segment_col = st.selectbox("Segment By (optional)", ["None"] + cat_cols)

# ── Prepare time series ──────────────────────────────────────────────────────
try:
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors="coerce")
except Exception:
    pass

df_clean = df[[date_col, sales_col]].dropna().copy()
df_clean = df_clean.sort_values(date_col)
df_clean[sales_col] = pd.to_numeric(df_clean[sales_col], errors="coerce")
ts = df_clean.groupby(date_col)[sales_col].sum().reset_index()
ts.columns = ["ds", "y"]

if len(ts) < 4:
    st.error("Not enough data points (need ≥ 4). Check your column selections.")
    st.stop()

# ─── KPI Row ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

total  = ts["y"].sum()
mean_s = ts["y"].mean()
peak   = ts["y"].max()
latest = ts["y"].iloc[-1]
prev   = ts["y"].iloc[-2] if len(ts) > 1 else latest
mom    = (latest - prev) / abs(prev) * 100 if prev != 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales",   f"${total:,.0f}")
c2.metric("Average / Period", f"${mean_s:,.0f}")
c3.metric("Peak Sales",    f"${peak:,.0f}")
c4.metric("Latest Period", f"${latest:,.0f}", delta=f"{mom:+.1f}% MoM")

# ─── Raw Data Preview ────────────────────────────────────────────────────────
with st.expander("🗂 Raw Data Preview", expanded=False):
    st.dataframe(df.head(200), use_container_width=True)
    st.caption(f"{len(df):,} rows · {len(df.columns)} columns")

# ─── Sales Trend ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Sales Trend</div>', unsafe_allow_html=True)

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=ts["ds"], y=ts["y"],
    mode="lines+markers",
    name="Actual Sales",
    line=dict(color=TEAL, width=2.5),
    marker=dict(size=5, color=TEAL),
    fill="tozeroy",
    fillcolor="rgba(100,255,218,0.06)",
))

# Rolling avg
if len(ts) >= 3:
    ts["rolling3"] = ts["y"].rolling(3, center=True).mean()
    fig_trend.add_trace(go.Scatter(
        x=ts["ds"], y=ts["rolling3"],
        mode="lines", name="3-Period MA",
        line=dict(color=GOLD, width=2, dash="dot"),
    ))

fig_trend.update_layout(**PLOTLY_LAYOUT, title="Historical Sales", height=350,
                         legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3561"))
st.plotly_chart(fig_trend, use_container_width=True)

# ─── Forecast ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Forecast</div>', unsafe_allow_html=True)

forecast_vals, fitted_vals, mae, rmse, r2 = build_forecast(ts["y"], forecast_periods, model_type)

# Build future dates
last_date = ts["ds"].iloc[-1]
freq_guess = pd.infer_freq(ts["ds"]) or "MS"
try:
    future_dates = pd.date_range(last_date, periods=forecast_periods + 1, freq=freq_guess)[1:]
except Exception:
    future_dates = pd.date_range(last_date, periods=forecast_periods + 1, freq="MS")[1:]

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=ts["ds"], y=ts["y"],
    mode="lines+markers", name="Actual",
    line=dict(color=TEAL, width=2),
    marker=dict(size=5),
))
fig_fc.add_trace(go.Scatter(
    x=ts["ds"], y=fitted_vals,
    mode="lines", name="Fitted",
    line=dict(color=PURPLE, width=1.5, dash="dot"),
))
fig_fc.add_trace(go.Scatter(
    x=future_dates, y=forecast_vals,
    mode="lines+markers", name=f"Forecast ({model_type})",
    line=dict(color=CORAL, width=2.5, dash="dash"),
    marker=dict(size=7, symbol="diamond", color=CORAL),
))

# Confidence band (±10%)
ci_upper = forecast_vals * 1.10
ci_lower = forecast_vals * 0.90
fig_fc.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(ci_upper) + list(ci_lower[::-1]),
    fill="toself", fillcolor="rgba(255,107,107,0.10)",
    line=dict(color="rgba(0,0,0,0)"),
    name="90% CI",
))

fig_fc.update_layout(**PLOTLY_LAYOUT, title="Sales Forecast", height=400,
                     legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3561"))
st.plotly_chart(fig_fc, use_container_width=True)

# ── Forecast table ───────────────────────────────────────────────────────────
fc_df = pd.DataFrame({"Period": future_dates, "Forecast": forecast_vals.round(2),
                       "Lower (90%)": ci_lower.round(2), "Upper (90%)": ci_upper.round(2)})
fc_df["Period"] = fc_df["Period"].dt.strftime("%Y-%m-%d")
st.dataframe(fc_df, use_container_width=True)

# ─── Model Performance ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("MAE",  f"${mae:,.0f}")
m2.metric("RMSE", f"${rmse:,.0f}")
m3.metric("R²",   f"{r2:.3f}")

# ─── Segment Analysis ────────────────────────────────────────────────────────
if segment_col != "None" and segment_col in df.columns:
    st.markdown(f'<div class="section-header">Sales by {segment_col}</div>', unsafe_allow_html=True)

    seg_df = df.groupby(segment_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False)
    seg_df.columns = ["Segment", "Total Sales"]

    col_bar, col_pie = st.columns(2)

    with col_bar:
        fig_bar = px.bar(seg_df, x="Segment", y="Total Sales",
                         color="Total Sales",
                         color_continuous_scale=[[0, "#2d3561"], [1, TEAL]],
                         title=f"Total Sales by {segment_col}")
        fig_bar.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False, height=320)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        fig_pie = px.pie(seg_df, names="Segment", values="Total Sales",
                         hole=0.45, title="Share of Sales",
                         color_discrete_sequence=px.colors.sequential.Teal)
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=320,
                               legend=dict(bgcolor="#1a1d2e"))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Trend by segment
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        seg_trend = df.groupby([date_col, segment_col])[sales_col].sum().reset_index()
        fig_seg = px.line(seg_trend, x=date_col, y=sales_col, color=segment_col,
                          title=f"Sales Trend by {segment_col}",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_seg.update_layout(**PLOTLY_LAYOUT, height=330,
                               legend=dict(bgcolor="#1a1d2e", bordercolor="#2d3561"))
        st.plotly_chart(fig_seg, use_container_width=True)
    except Exception:
        pass

# ─── Distribution ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Distribution & Seasonality</div>', unsafe_allow_html=True)

col_hist, col_box = st.columns(2)

with col_hist:
    fig_h = px.histogram(ts, x="y", nbins=20, title="Sales Distribution",
                         color_discrete_sequence=[PURPLE])
    fig_h.update_layout(**PLOTLY_LAYOUT, height=300)
    st.plotly_chart(fig_h, use_container_width=True)

with col_box:
    ts["month"] = pd.to_datetime(ts["ds"]).dt.month_name()
    ts["month_num"] = pd.to_datetime(ts["ds"]).dt.month
    monthly = ts.sort_values("month_num")
    fig_box = px.box(monthly, x="month", y="y", title="Monthly Distribution",
                     color_discrete_sequence=[TEAL],
                     category_orders={"month": ["January","February","March","April","May","June",
                                                  "July","August","September","October","November","December"]})
    fig_box.update_layout(**PLOTLY_LAYOUT, height=300)
    st.plotly_chart(fig_box, use_container_width=True)

# ─── Growth Analysis ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Growth Analysis</div>', unsafe_allow_html=True)

ts["pct_change"] = ts["y"].pct_change() * 100

fig_growth = go.Figure()
colors = [TEAL if v >= 0 else CORAL for v in ts["pct_change"].fillna(0)]
fig_growth.add_trace(go.Bar(
    x=ts["ds"], y=ts["pct_change"],
    marker_color=colors, name="Period-over-Period Growth %",
))
fig_growth.add_hline(y=0, line_color="#2d3561", line_width=1)
fig_growth.update_layout(**PLOTLY_LAYOUT, title="Period-over-Period Growth (%)", height=300)
st.plotly_chart(fig_growth, use_container_width=True)

# ─── Insights ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Auto-Generated Insights</div>', unsafe_allow_html=True)

growth_trend = "upward 📈" if ts["y"].iloc[-1] > ts["y"].iloc[0] else "downward 📉"
best_period  = ts.loc[ts["y"].idxmax(), "ds"].strftime("%b %Y")
worst_period = ts.loc[ts["y"].idxmin(), "ds"].strftime("%b %Y")
avg_growth   = ts["pct_change"].mean()
total_fc     = forecast_vals.sum()

insights = [
    f"📊 Overall sales trend is <strong>{growth_trend}</strong> across {len(ts)} periods.",
    f"🏆 Best performing period: <strong>{best_period}</strong> (${ts['y'].max():,.0f}).",
    f"⚠️ Lowest sales period: <strong>{worst_period}</strong> (${ts['y'].min():,.0f}).",
    f"📈 Average period-over-period growth: <strong>{avg_growth:+.1f}%</strong>.",
    f"🔮 Forecasted total for next {forecast_periods} periods: <strong>${total_fc:,.0f}</strong> (±10% CI).",
    f"🎯 Model R² of <strong>{r2:.3f}</strong> — {'good fit' if r2 > 0.7 else 'moderate fit; consider more data'} using {model_type}.",
]
for ins in insights:
    st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:12px;padding:16px 0;'>"
    "Sales Forecasting Dashboard · Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True,
)
