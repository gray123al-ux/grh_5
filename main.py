from pathlib import Path

main_py = r'''import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]

st.title(APP_TITLE)
st.caption("영화의 개봉 규모와 흥행 지속성을 바탕으로 비슷한 영화끼리 묶어 봅니다.")

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

    for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 상용로그는 양수에서만 정의되므로 0 이하 값은 결측 처리
    df["log_first_scrn"] = df["first_scrn"].map(
        lambda value: math.log10(value)
        if pd.notna(value) and value > 0
        else float("nan")
    )
    df["log_total_audi"] = df["total_audi"].map(
        lambda value: math.log10(value)
        if pd.notna(value) and value > 0
        else float("nan")
    )

    # 롱런 지수 = 누적 관객 / 첫 주 관객, 최대 20
    df["longrun_index"] = (df["total_audi"] / df["first_week_audi"]).clip(upper=20)

    # 네 속성 중 결측이 있거나 첫 주 관객이 0 이하인 영화 제외
    usable = df.dropna(
        subset=FEATURE_COLS + ["first_week_audi", "movieNm"]
    ).copy()
    usable = usable[
        (usable["first_week_audi"] > 0)
        & (usable["first_scrn"] > 0)
        & (usable["total_audi"] > 0)
    ].copy()

    return usable, total_movies


try:
    usable_df, total_movies = load_data(DATA_URL)
except Exception as error:
    st.error("데이터를 불러오거나 처리하지 못했습니다. 데이터 주소와 CSV 열 이름을 확인해 주세요.")
    st.exception(error)
    st.stop()

st.markdown(
    f"**전체 영화 {total_movies:,}편 · 분석 가능한 영화 {len(usable_df):,}편**"
)

if len(usable_df) < 7:
    st.warning("묶음 수 1~7을 비교하려면 분석 가능한 영화가 최소 7편 필요합니다.")
    st.stop()

# --------------------------------------------------
# 1. 분류 속성 및 묶음 수 선택
# --------------------------------------------------
st.subheader("1. 분류 설정")
left, right = st.columns([2, 1])

with left:
    selected_features = st.multiselect(
        "묶는 데 사용할 속성 (두 개 이상 선택)",
        options=FEATURE_LABELS,
        default=FEATURE_LABELS,
        help="선택한 속성만 표준화한 뒤 K-평균 분류와 평가에 사용합니다.",
    )

with right:
    cluster_count = st.selectbox(
        "묶음 수",
        options=[2, 3, 4, 5, 6, 7],
        index=1,
        help="분류할 묶음 수를 2~7개 중에서 선택합니다.",
    )

if len(selected_features) < 2:
    st.info("분류를 진행하려면 속성을 두 개 이상 선택하세요.")
    st.stop()

selected_cols = [FEATURES[label] for label in selected_features]
analysis_df = usable_df.dropna(subset=selected_cols).copy()

if len(analysis_df) < 7:
    st.warning("선택한 속성에서 분석 가능한 영화가 7편 미만이라 비교 그래프를 만들 수 없습니다.")
    st.stop()

# 선택한 속성 표준화: 엘보 그래프와 최종 분류에 같은 데이터를 사용
scaler = StandardScaler()
scaled_values = scaler.fit_transform(analysis_df[selected_cols])

# --------------------------------------------------
# 2. K=1~7 중심 거리 제곱합
# --------------------------------------------------
inertia_rows = []
for k in range(1, 8):
    elbow_model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    elbow_model.fit(scaled_values)
    inertia_rows.append({
        "묶음 수": k,
        "중심 거리 제곱합": float(elbow_model.inertia_),
    })

inertia_df = pd.DataFrame(inertia_rows)

# --------------------------------------------------
# 3. 선택한 묶음 수로 K-평균
# --------------------------------------------------
kmeans = KMeans(
    n_clusters=cluster_count,
    random_state=RANDOM_STATE,
    n_init=10,
)
analysis_df["cluster_raw"] = kmeans.fit_predict(scaled_values)

# 누적 관객 평균이 높은 묶음부터 ㉮, ㉯, ... 부여
cluster_means = analysis_df.groupby("cluster_raw")["total_audi"].mean()
sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()
cluster_name_map = {
    raw_cluster: CLUSTER_SYMBOLS[index]
    for index, raw_cluster in enumerate(sorted_clusters)
}
analysis_df["묶음"] = analysis_df["cluster_raw"].map(cluster_name_map)
analysis_df["영화 제목"] = analysis_df["movieNm"].fillna("제목 없음").astype(str)
active_symbols = CLUSTER_SYMBOLS[:cluster_count]

# 실루엣 점수는 K=2~7에서 계산 가능
try:
    silhouette = silhouette_score(scaled_values, analysis_df["cluster_raw"])
except Exception:
    silhouette = float("nan")

st.caption(
    f"분류 방식: 표준화 후 K-평균 · 묶음 수 {cluster_count}개 · 난수 시드 {RANDOM_STATE}"
)

if pd.notna(silhouette):
    st.markdown(
        f"**현재 묶음 수 {cluster_count}개의 실루엣 점수: {silhouette:.3f}**"
        "  \n실루엣 점수는 -1~1 범위이며, 1에 가까울수록 묶음이 뚜렷하게 구분되는 경향이 있습니다."
    )
else:
    st.info("현재 데이터에서는 실루엣 점수를 계산할 수 없습니다.")

# --------------------------------------------------
# 4. 엘보 그래프와 감소량 표
# --------------------------------------------------
st.subheader("2. 묶음 수에 따른 중심 거리 제곱합")
st.write(
    "각 영화가 자기 묶음의 중심에서 떨어진 거리의 제곱을 모두 더한 값입니다. "
    "묶음 수가 늘어날 때 값이 얼마나 줄어드는지 살펴보세요."
)

fig_elbow = px.line(
    inertia_df,
    x="묶음 수",
    y="중심 거리 제곱합",
    markers=True,
    labels={
        "묶음 수": "묶음 수 (K)",
        "중심 거리 제곱합": "중심 거리 제곱합",
    },
    title="엘보 그래프: 묶음 수와 중심 거리 제곱합",
)
fig_elbow.add_vline(
    x=cluster_count,
    line_dash="dash",
    line_width=2,
    annotation_text=f"현재 선택: K={cluster_count}",
    annotation_position="top right",
)
fig_elbow.update_layout(height=430, xaxis=dict(dtick=1), showlegend=False)
st.plotly_chart(fig_elbow, use_container_width=True)

inertia_table = inertia_df.copy()
inertia_table["중심 거리 제곱합"] = inertia_table["중심 거리 제곱합"].map(
    lambda value: f"{value:,.3f}"
)

decreases = [None]
for i in range(1, len(inertia_df)):
    previous_value = inertia_df.loc[i - 1, "중심 거리 제곱합"]
    current_value = inertia_df.loc[i, "중심 거리 제곱합"]
    decreases.append(previous_value - current_value)

inertia_table["앞 묶음 수 대비 감소량"] = [
    "" if value is None else f"{value:,.3f}" for value in decreases
]
st.dataframe(inertia_table, use_container_width=True, hide_index=True)

# --------------------------------------------------
# 5. 2차원 산점도
# --------------------------------------------------
st.subheader("3. 2차원 산점도")
col_x, col_y = st.columns(2)

with col_x:
    x_label = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        key="scatter2d_x",
    )

with col_y:
    y_label = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=1,
        key="scatter2d_y",
    )

fig2d = px.scatter(
    analysis_df,
    x=FEATURES[x_label],
    y=FEATURES[y_label],
    color="묶음",
    category_orders={"묶음": active_symbols},
    hover_name="영화 제목",
    hover_data={
        "묶음": True,
        FEATURES[x_label]: ":.3f",
        FEATURES[y_label]: ":.3f",
        "movieNm": False,
        "cluster_raw": False,
    },
    labels={
        FEATURES[x_label]: x_label,
        FEATURES[y_label]: y_label,
        "묶음": "영화 유형",
    },
    title=f"{x_label} × {y_label}",
)
fig2d.update_traces(marker={"size": 8, "opacity": 0.78})
fig2d.update_layout(legend_title_text="영화 유형", height=520)
st.plotly_chart(fig2d, use_container_width=True)

# --------------------------------------------------
# 6. 3차원 산점도
# --------------------------------------------------
st.subheader("4. 3차원 산점도")

if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 위에서 분류 속성을 세 개 이상 선택하세요.")
else:
    col_x3, col_y3, col_z3 = st.columns(3)

    with col_x3:
        x3_label = st.selectbox(
            "X축 속성", selected_features, index=0, key="scatter3d_x"
        )

    with col_y3:
        y3_label = st.selectbox(
            "Y축 속성", selected_features, index=1, key="scatter3d_y"
        )

    with col_z3:
        z3_label = st.selectbox(
            "Z축 속성", selected_features, index=2, key="scatter3d_z"
        )

    fig3d = px.scatter_3d(
        analysis_df,
        x=FEATURES[x3_label],
        y=FEATURES[y3_label],
        z=FEATURES[z3_label],
        color="묶음",
        category_orders={"묶음": active_symbols},
        hover_name="영화 제목",
        hover_data={
            "묶음": True,
            FEATURES[x3_label]: ":.3f",
            FEATURES[y3_label]: ":.3f",
            FEATURES[z3_label]: ":.3f",
            "movieNm": False,
            "cluster_raw": False,
        },
        labels={
            FEATURES[x3_label]: x3_label,
            FEATURES[y3_label]: y3_label,
            FEATURES[z3_label]: z3_label,
            "묶음": "영화 유형",
        },
        title="마우스로 회전하며 세 속성의 관계를 살펴보세요",
    )
    fig3d.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3d.update_layout(
        height=650,
        legend_title_text="영화 유형",
        scene={
            "xaxis_title": x3_label,
            "yaxis_title": y3_label,
            "zaxis_title": z3_label,
        },
    )
    st.plotly_chart(fig3d, use_container_width=True)

# --------------------------------------------------
# 7. 묶음별 평균과 편수
# --------------------------------------------------
st.subheader("5. 묶음별 요약")
summary_rows = []

for symbol in active_symbols:
    group = analysis_df[analysis_df["묶음"] == symbol]
    if group.empty:
        continue

    summary_rows.append({
        "묶음": symbol,
        "영화 편수": len(group),
        "스크린 수 평균": group["first_scrn"].mean(),
        "누적 관객 평균": group["total_audi"].mean(),
        "10위권 일수 평균": group["days_in_top10"].mean(),
        "롱런 지수 평균": group["longrun_index"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_display = summary_df.copy()

if not summary_display.empty:
    summary_display["영화 편수"] = summary_display["영화 편수"].map(
        lambda value: f"{value:,}"
    )
    summary_display["스크린 수 평균"] = summary_display["스크린 수 평균"].map(
        lambda value: f"{value:,.1f}"
    )
    summary_display["누적 관객 평균"] = summary_display["누적 관객 평균"].map(
        lambda value: f"{value:,.0f}"
    )
    summary_display["10위권 일수 평균"] = summary_display["10위권 일수 평균"].map(
        lambda value: f"{value:,.1f}"
    )
    summary_display["롱런 지수 평균"] = summary_display["롱런 지수 평균"].map(
        lambda value: f"{value:,.2f}"
    )

st.dataframe(summary_display, use_container_width=True, hide_index=True)

# --------------------------------------------------
# 8. 묶음별 누적 관객 상위 5편
# --------------------------------------------------
st.subheader("6. 묶음별 누적 관객 상위 영화")

# 최대 7개 묶음을 보기 좋게 최대 4열로 배치하고, 남는 묶음은 다음 행에 표시
for start in range(0, cluster_count, 4):
    row_symbols = active_symbols[start:start + 4]
    columns = st.columns(len(row_symbols))

    for column, symbol in zip(columns, row_symbols):
        group = (
            analysis_df[analysis_df["묶음"] == symbol]
            .sort_values("total_audi", ascending=False)
            .head(5)
        )

        with column:
            st.markdown(f"#### {symbol} 묶음")
            if group.empty:
                st.write("해당 묶음에 영화가 없습니다.")
            else:
                for rank, (_, movie) in enumerate(group.iterrows(), start=1):
                    st.markdown(f"{rank}. **{movie['영화 제목']}**")
                    st.caption(f"누적 관객 {movie['total_audi']:,.0f}명")

st.divider()
st.caption(
    "참고: K-평균은 선택한 속성의 거리와 분포를 바탕으로 묶습니다. "
    "묶음은 정답이나 영화의 절대적인 품질을 뜻하지 않으며, 속성 선택과 묶음 수에 따라 달라질 수 있습니다."
)
'''

requirements = """streamlit
pandas
plotly
scikit-learn
"""

# Syntax-check before writing files
compile(main_py, "main.py", "exec")
Path("/mnt/data/main.py").write_text(main_py, encoding="utf-8")
Path("/mnt/data/requirements.txt").write_text(requirements, encoding="utf-8")
print("Fixed and syntax-checked /mnt/data/main.py")
