import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import os

# 📂 데이터 로드
file_path = os.path.join("data", "상품별공급량_MJ.xlsx")
df = pd.read_excel(file_path, sheet_name="데이터")

# ❌ 불필요 열 제거
df = df.drop(columns=["총합계", "비교(V-W)"])
product_cols = df.columns[4:]

# 🎛️ 사이드바
st.sidebar.header("🔧 설정")
available_years = sorted(df["연"].unique())
selected_years = st.sidebar.multiselect("학습 연도 선택", options=available_years, default=available_years)
selected_product = st.sidebar.selectbox("상품 선택", options=product_cols)

# 🎯 데이터 필터링
filtered_df = df[df["연"].isin(selected_years)].copy()

# 🧼 해당 상품만 전처리 (NaN, 0 제거)
filtered_df = filtered_df[(~filtered_df[selected_product].isna()) & (filtered_df[selected_product] != 0)]

# 🧠 모델 학습
X = filtered_df[["평균기온"]]
y = filtered_df[selected_product]
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)
r2 = r2_score(y, y_pred)

# 📈 시각화
x_range = np.linspace(X["평균기온"].min(), X["평균기온"].max(), 300)
x_range_poly = poly.transform(x_range.reshape(-1, 1))
y_range_pred = model.predict(x_range_poly)

fig = go.Figure()
fig.add_trace(go.Scatter(x=X["평균기온"], y=y, mode="markers", name="실제값", marker=dict(size=6)))
fig.add_trace(go.Scatter(x=x_range, y=y_range_pred, mode="lines", name="모델 예측 (3차)", line=dict(width=2)))

fig.update_layout(
    title=f"{selected_product} 공급량 vs 평균기온 (R² = {r2:.4f})",
    xaxis_title="평균기온 (℃)",
    yaxis_title="공급량",
    template="plotly_white",
    height=600
)

# ✅ 출력
st.title("📊 평균기온과 상품별 공급량 관계 시각화")
st.plotly_chart(fig, use_container_width=True)
