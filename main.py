import streamlit as st
import pandas as pd
import numpy as np
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

# 🎛️ 사이드바 설정
st.sidebar.header("🔧 설정")
available_years = sorted(df["연"].unique())
selected_years = st.sidebar.multiselect("학습 데이터 연도 선택", options=available_years, default=available_years)

# ✅ 예측 상품 디폴트 순서
ordered_default_products = [
    "개별난방용", "중앙난방용", "자가열전용", "일반용(2)",
    "업무난방용", "냉난방용", "주한미군", "총공급량"
]
selected_products = st.sidebar.multiselect(
    "예측할 상품 선택",
    options=product_cols,
    default=[p for p in ordered_default_products if p in product_cols]
)

# 📅 예측 기간 설정
start_date = st.sidebar.date_input("예측 시작 월", value=pd.to_datetime("2026-01-01"))
end_date = st.sidebar.date_input("예측 종료 월", value=pd.to_datetime("2026-12-01"))

if start_date > end_date:
    st.sidebar.error("❌ 시작 월은 종료 월보다 이전이어야 합니다.")

# ⏳ 학습 데이터 필터링 (연도만 적용)
filtered_df = df[df["연"].isin(selected_years)]

# 🧠 모델 훈련 함수 (전체 상품 대상, R² 포함)
def train_models(data, products):
    models = {}
    r2_scores = {}
    for product in products:
        product_data = data[(~data[product].isna()) & (data[product] != 0)]
        if product_data.empty:
            continue  # 데이터가 없으면 생략
        X = product_data[["평균기온"]]
        y = product_data[product]
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        models[product] = (model, poly)
        r2_scores[product] = r2
    return models, r2_scores

# 🔁 전체 상품 모델 훈련
models, r2_scores = train_models(filtered_df, product_cols)

# 📊 전체 상품 R² 요약 테이블
if r2_scores:
    r2_df = pd.DataFrame([
        {
            "상품명": k,
            "R²": round(v, 4),
            "예측 적합도": (
                "✅ 매우 높음" if v >= 0.85 else
                "✅ 높음" if v >= 0.7 else
                "⚠️ 보통" if v >= 0.5 else
                "❌ 낮음" if v >= 0.3 else
                "❌ 매우 낮음"
            )
        }
        for k, v in r2_scores.items()
    ]).sort_values(by="R²", ascending=False)

    with st.expander("📈 전체 상품의 R² 및 예측 적합도 요약", expanded=False):
        st.dataframe(r2_df, use_container_width=True)

# 📆 예측 대상 월 리스트 생성
def generate_month_list(start, end):
    return pd.date_range(start=start, end=end, freq="MS")

month_list = generate_month_list(start_date, end_date)

# 📥 입력 템플릿
st.title("📦 월별 평균기온 기반 상품별 공급량 예측")
st.write("아래 표에서 예측 기간의 평균기온(℃)을 입력하세요.")

input_template = pd.DataFrame({
    "날짜": month_list,
    "평균기온": [0.0] * len(month_list)
})
input_df = st.data_editor(input_template, use_container_width=True, num_rows="fixed", key="temperature_input")

# ▶️ 예측 실행
if not input_df.empty and selected_products:
    results = input_df.copy()
    for product in selected_products:
        if product not in models:
            continue  # 모델이 없는 상품은 생략
        model, poly = models[product]
        X_input = poly.transform(input_df[["평균기온"]])
        preds = model.predict(X_input)
        results[product + "_예측공급량"] = preds.round(0).astype(int)

    st.subheader("📊 예측 결과")
    st.dataframe(results, use_container_width=True)
