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
    x3_col, y3_col, z3_col = st.columns(3)
    with x3_col:
        x3_label = st.selectbox(
            "X축 속성", selected_features, index=0, key="scatter3d_x"
        )
    with y3_col:
        y3_label = st.selectbox(
            "Y축 속성", selected_features, index=1, key="scatter3d_y"
        )
    with z3_col:
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
# 7. 묶음별 평균 및 편수
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
summary_display["영화 편수"] = summary_display["영화 편수"].map(lambda value: f"{value:,}")
summary_display["스크린 수 평균"] = summary_display["스크린 수 평균"].map(lambda value: f"{value:,.1f}")
summary_display["누적 관객 평균"] = summary_display["누적 관객 평균"].map(lambda value: f"{value:,.0f}")
summary_display["10위권 일수 평균"] = summary_display["10위권 일수 평균"].map(lambda value: f"{value:,.1f}")
summary_display["롱런 지수 평균"] = summary_display["롱런 지수 평균"].map(lambda value: f"{value:,.2f}")
st.dataframe(summary_display, use_container_width=True, hide_index=True)

# --------------------------------------------------
# 8. 묶음별 누적 관객 상위 5편
# --------------------------------------------------
st.subheader("6. 묶음별 누적 관객 상위 영화")

result_columns = st.columns(min(cluster_count, 4))
for index, symbol in enumerate(active_symbols):
    # 4개 이상인 경우에도 모든 묶음을 표시하되, 열을 반복 사용
    column = result_columns[index % len(result_columns)]
    group = analysis_df[analysis_df["묶음"] == symbol].sort_values(
        "total_audi", ascending=False
    ).head(5)

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
