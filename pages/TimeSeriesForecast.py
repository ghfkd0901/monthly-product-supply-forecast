import streamlit as st
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# 📂 데이터 로드
file_path = os.path.join("data", "상품별공급량_MJ.xlsx")
df = pd.read_excel(file_path, sheet_name="데이터")

# 🗓️ 날짜 처리
df["날짜"] = pd.to_datetime(df[["연", "월"]].rename(columns={"연": "year", "월": "month"}).assign(day=1))

# 🎛️ 사이드바 설정
st.sidebar.header("🔧 설정")
available_years = sorted(df["연"].dropna().unique())
selected_years = st.sidebar.multiselect("학습 데이터 연도 선택", options=available_years, default=available_years)
forecast_months = st.sidebar.slider("예측 개월 수", min_value=3, max_value=24, value=12)
model_type = st.sidebar.selectbox("예측 모델", ["SARIMA", "Prophet"])

# 예측 대상 상품 선택
default_products = ["취사용", "일반용(1)", "산업용"]
product_cols = df.select_dtypes(include=[np.number]).columns.difference(["연", "월", "총합계", "비교(V-W)"]).tolist()
selected_products = st.sidebar.multiselect("예측할 상품 선택", options=product_cols, default=[p for p in default_products if p in product_cols])

# 📈 결과 담을 프레임
result_df = pd.DataFrame()
future_dates = pd.date_range(start=df["날짜"].max() + pd.DateOffset(months=1), periods=forecast_months, freq="MS")
result_df["날짜"] = future_dates

# 📊 그래프
st.title("📈 예측 공급량 시계열 그래프")
fig = go.Figure()

for product in selected_products:
    df_filtered = df[df["연"].isin(selected_years)].copy()
    series = df_filtered[["날짜", product]].dropna().sort_values("날짜")

    if len(series) < 24:
        continue  # 학습 데이터가 부족하면 skip

    # 과거 그래프 그리기 (파란색)
    fig.add_trace(go.Scatter(
        x=series["날짜"],
        y=series[product],
        mode="lines+markers",
        name=f"{product} (과거)",
    ))

    # 예측
    if model_type == "SARIMA":
        series.set_index("날짜", inplace=True)
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
        forecast = model.forecast(steps=forecast_months)
        result_df[f"{product}_예측공급량"] = forecast.values

    elif model_type == "Prophet":
        prophet_df = series.reset_index() if isinstance(series.index, pd.DatetimeIndex) else series.copy()
        prophet_df.columns = ["ds", "y"]
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_months, freq="MS")
        forecast = model.predict(future)[["ds", "yhat"]].tail(forecast_months)
        result_df[f"{product}_예측공급량"] = forecast["yhat"].values

    # 예측 그래프 추가 (빨간색)
    fig.add_trace(go.Scatter(
        x=result_df["날짜"],
        y=result_df[f"{product}_예측공급량"],
        mode="lines+markers",
        name=f"{product} (예측)",
        line=dict(color="red", dash="dot")
    ))

# 전체 그래프 스타일
fig.update_layout(
    xaxis_title="날짜",
    yaxis_title="공급량(MJ)",
    height=600,
    template="plotly_white"
)

# 시각화 우선 표시
st.plotly_chart(fig, use_container_width=True)

# 예측 결과 테이블 아래에 출력
st.subheader("📋 예측 결과 표")
st.dataframe(result_df, use_container_width=True)
