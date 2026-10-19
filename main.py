import math

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

APP_TITLE = "🎬 영화 유형 나누기"
DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
RANDOM_STATE = 42

FEATURES = {
    "스크린 수 (로그)": "log_first_scrn",
    "누적 관객 (로그)": "log_total_audi",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "longrun_index",
}
FEATURE_LABELS = list(FEATURES.keys())
FEATURE_COLS = list(FEATURES.values())

# 요청한 기호 순서: 묶음 수가 늘어날 때 이어서 사용
CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

st.title(APP_TITLE)
st.caption(
    "영화의 개봉 규모와 흥행 지속성을 바탕으로 비슷한 영화끼리 묶어 봅니다."
)

# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data(show_spinner="영화 데이터를 불러오는 중...")
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")

    required_columns = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing_columns))

    total_movies = len(df)

    numeric_columns = [
        "first_scrn", "first_week_audi", "total_audi", "days_in_top10"
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 스크린 수와 누적 관객은 상용로그로 변환한다.
    # 로그는 양수에 대해서만 정의되므로 0 이하 값은 결측 처리한다.
    df["log_first_scrn"] = df["first_scrn"].map(
        lambda value: math.log10(value) if pd.notna(value) and value > 0 else float("nan")
    )
    df["log_total_audi"] = df["total_audi"].map(
        lambda value: math.log10(value) if pd.notna(value) and value > 0 else float("nan")
    )

    # 누적 관객 / 첫 주 관객. 20을 넘는 값은 20으로 제한한다.
    df["longrun_index"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)

    # 네 속성 중 결측이 있거나 첫 주 관객이 0 이하인 영화는 제외한다.
    usable = df.dropna(
        subset=FEATURE_COLS + ["first_week_audi", "movieNm"]
    ).copy()
    usable = usable[
        (usable["first_week_audi"] > 0)
        & (usable["first_scrn"] > 0)
        & (usable["total_audi"] > 0)
    ].copy()

    return df, usable, total_movies


try:
    original_df, usable_df, total_movies = load_data(DATA_URL)
except Exception as error:
    st.error("데이터를 불러오거나 처리하지 못했습니다. 데이터 주소와 CSV 열 이름을 확인해 주세요.")
    st.exception(error)
    st.stop()
