import streamlit as st
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import plotly.graph_objects as go
import warnings
import hashlib
warnings.filterwarnings("ignore")

# ✅ 캐시된 데이터 로드
@st.cache_data
def load_data():
    file_path = os.path.join("data", "상품별공급량_MJ.xlsx")
    return pd.read_excel(file_path, sheet_name="데이터")

# ✅ 캐시 키 생성
def make_cache_key(model_type, product, years, forecast_months):
    key_str = f"{model_type}_{product}_{','.join(map(str, years))}_{forecast_months}"
    return hashlib.md5(key_str.encode()).hexdigest()

# ✅ 세션 상태 캐시 초기화
if "model_cache" not in st.session_state:
    st.session_state.model_cache = {}

# 📂 데이터 로드 및 날짜 처리
df = load_data()
df["날짜"] = pd.to_datetime(df[["연", "월"]].rename(columns={"연": "year", "월": "month"}).assign(day=1))

# 🎛️ 사이드바 설정
st.sidebar.header("🔧 설정")
available_years = sorted(df["연"].dropna().unique())
selected_years = st.sidebar.multiselect("학습 데이터 연도 선택", options=available_years, default=available_years)
forecast_months = st.sidebar.slider("예측 개월 수", min_value=3, max_value=24, value=12)
model_type = st.sidebar.selectbox("예측 모델", ["SARIMA", "Prophet", "Holt-Winters"])

# 예측 대상 상품 선택
product_cols = df.select_dtypes(include=[np.number]).columns.difference(["연", "월", "총합계", "비교(V-W)"]).tolist()
selected_products = st.sidebar.multiselect("예측할 상품 선택", options=product_cols, default=["일반용(1)"])

# 안내
st.markdown("""
⏳ **모델 학습에는 최대 2분 정도 소요될 수 있습니다.**  
☕ 잠시 커피 한잔 하고 오시면 예측 결과가 완성되어 있을 거예요!
""")

# 📅 미래 날짜 생성
future_dates = pd.date_range(start=df["날짜"].max() + pd.DateOffset(months=1), periods=forecast_months, freq="MS")
result_df = pd.DataFrame({"날짜": future_dates})

# 🔁 상품별 학습 및 예측
for product in selected_products:
    df_filtered = df[df["연"].isin(selected_years)].copy()
    series = df_filtered[["날짜", product]].dropna().sort_values("날짜")

    if len(series) < 24:
        st.warning(f"🔍 `{product}` 데이터가 부족하여 예측이 생략되었습니다.")
        continue

    cache_key = make_cache_key(model_type, product, selected_years, forecast_months)

    # 모델 캐시 확인 또는 학습
    if cache_key in st.session_state.model_cache:
        forecast = st.session_state.model_cache[cache_key]
    else:
        forecast = None
        with st.spinner(f"{product} {model_type} 모델 학습 중..."):
            try:
                if model_type == "SARIMA":
                    series.set_index("날짜", inplace=True)
                    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                    forecast = model.forecast(steps=forecast_months).tolist()

                elif model_type == "Prophet":
                    prophet_df = series.rename(columns={"날짜": "ds", product: "y"})
                    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    model.fit(prophet_df)
                    future = model.make_future_dataframe(periods=forecast_months, freq="MS")
                    forecast_df = model.predict(future)[["ds", "yhat"]].tail(forecast_months)
                    forecast = forecast_df["yhat"].tolist()

                elif model_type == "Holt-Winters":
                    series_hw = series.set_index("날짜")[product]
                    model = ExponentialSmoothing(series_hw, seasonal='add', seasonal_periods=12).fit()
                    forecast = model.forecast(forecast_months).tolist()

                st.session_state.model_cache[cache_key] = forecast

            except Exception as e:
                st.error(f"❌ {product} 모델 학습 오류: {e}")
                continue

    # 예측 결과 저장
    if forecast:
        result_df[f"{product}_예측공급량"] = forecast

# ✅ 예측 결과 표 먼저 출력
st.subheader("📋 예측 결과 표")
st.dataframe(result_df, use_container_width=True)

# ✅ 시계열 그래프 시각화
st.title("📈 예측 공급량 시계열 그래프")
fig = go.Figure()

for product in selected_products:
    df_filtered = df[df["연"].isin(selected_years)].copy()
    series = df_filtered[["날짜", product]].dropna().sort_values("날짜")

    if len(series) < 24:
        continue

    # 과거 그래프
    fig.add_trace(go.Scatter(x=series["날짜"], y=series[product], mode="lines+markers", name=f"{product} (과거)"))

    # 예측 그래프
    forecast_col = f"{product}_예측공급량"
    if forecast_col in result_df.columns:
        fig.add_trace(go.Scatter(
            x=result_df["날짜"],
            y=result_df[forecast_col],
            mode="lines+markers",
            name=f"{product} (예측)",
            line=dict(color="red", dash="dot")
        ))

fig.update_layout(
    xaxis_title="날짜",
    yaxis_title="공급량(MJ)",
    height=600,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
