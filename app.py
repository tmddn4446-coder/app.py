import streamlit as st
import pandas as pd
import numpy as np

# 페이지 설정
st.set_page_config(page_title="AHP 통합 분석 시스템", layout="wide")

st.title("🪖 교육훈련 기간 체계 개선 AHP 설문 (1~3계층 통합)")
st.markdown("""
이 시스템은 **[1계층: 대분류] → [2계층: 하위항목] → [3계층: 180일형 vs 70일형]** 순서로 
쌍대비교를 수행하여 최종 가중치를 도출합니다.
""")

# ---------------------------------------------------------
# [공통 함수] 5점 척도 -> AHP 수치 변환 및 가중치 계산
# ---------------------------------------------------------
# 5점 척도 라벨
scale_labels = ["A 매우 중요(5)", "A 중요(3)", "동일(1)", "B 중요(3)", "B 매우 중요(5)"]
# 척도에 따른 AHP 점수 매핑 (A 기준)
scale_values = {
    "A 매우 중요(5)": 5.0,
    "A 중요(3)": 3.0,
    "동일(1)": 1.0,
    "B 중요(3)": 1/3.0,
    "B 매우 중요(5)": 1/5.0
}

def calculate_ahp_weights(matrix):
    """기하평균법을 이용한 가중치 계산"""
    n = matrix.shape[0]
    geometric_means = np.prod(matrix, axis=1) ** (1/n)
    weights = geometric_means / np.sum(geometric_means)
    return weights

def pairwise_input(label, item_a, item_b, key):
    """쌍대비교 슬라이더 UI"""
    st.write(f"**{item_a}** vs **{item_b}**")
    choice = st.select_slider(
        label, options=scale_labels, value="동일(1)", key=key, label_visibility="collapsed"
    )
    return scale_values[choice]

# 데이터 구조 정의
hierarchy = {
    "전투기술": ["전시물자 군수지원", "개인화기 사격", "편제장비 운용"],
    "작계시행": ["지휘통제기구 훈련", "작계지역 지형정찰", "증·창설 절차 숙달"],
    "교관능력": ["전시교관 자격인증평가", "공용화기 집체교육", "병진급개인기본훈련평가"]
}
l1_items = list(hierarchy.keys())

# ---------------------------------------------------------
# UI 구성: 탭(Tab)으로 계층 분리
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["1계층: 대분류", "2계층: 하위항목", "3계층: 대안비교", "🏆 최종 결과"])

# 글로벌 변수로 가중치 저장
if 'w_l1' not in st.session_state: st.session_state['w_l1'] = {}
if 'w_l2' not in st.session_state: st.session_state['w_l2'] = {}
if 'scores_l3' not in st.session_state: st.session_state['scores_l3'] = {}

# --- [Tab 1] 1계층 쌍대비교 ---
with tab1:
    st.header("1. 대분류 중요도 평가")
    st.info("각 대분류 간의 중요도를 비교해주세요.")
    
    col1, col2 = st.columns(2)
    # 3개 항목이므로 3번의 비교 필요 (A-B, A-C, B-C)
    with col1:
        comp_1_2 = pairwise_input("1vs2", l1_items[0], l1_items[1], "l1_1") # 전투 vs 작계
        comp_1_3 = pairwise_input("1vs3", l1_items[0], l1_items[2], "l1_2") # 전투 vs 교관
        comp_2_3 = pairwise_input("2vs3", l1_items[1], l1_items[2], "l1_3") # 작계 vs 교관

    # 행렬 생성 및 계산
    mat_l1 = np.array([
        [1.0, comp_1_2, comp_1_3],
        [1/comp_1_2, 1.0, comp_2_3],
        [1/comp_1_3, 1/comp_2_3, 1.0]
    ])
    
    w_l1 = calculate_ahp_weights(mat_l1)
    st.session_state['w_l1'] = dict(zip(l1_items, w_l1))
    
    with col2:
        st.subheader("실시간 결과 (가중치)")
        st.bar_chart(pd.Series(st.session_state['w_l1']))

# --- [Tab 2] 2계층 쌍대비교 ---
with tab2:
    st.header("2. 하위 훈련항목 중요도 평가")
    st.info("각 대분류 내에서 하위 항목들의 중요도를 비교해주세요.")
    
    local_weights_l2 = {} # 지역 가중치 저장
    
    cols = st.columns(3)
    
    idx = 0
    for main_cat in l1_items:
        sub_items = hierarchy[main_cat]
        with cols[idx]:
            st.subheader(f"📌 {main_cat}")
            # 3개 항목 비교
            v1 = pairwise_input(f"{main_cat}_1", sub_items[0], sub_items[1], f"l2_{main_cat}_1")
            v2 = pairwise_input(f"{main_cat}_2", sub_items[0], sub_items[2], f"l2_{main_cat}_2")
            v3 = pairwise_input(f"{main_cat}_3", sub_items[1], sub_items[2], f"l2_{main_cat}_3")
            
            mat_sub = np.array([
                [1.0, v1, v2],
                [1/v1, 1.0, v3],
                [1/v2, 1/v3, 1.0]
            ])
            w_sub = calculate_ahp_weights(mat_sub)
            
            # 딕셔너리에 저장
            for i, item in enumerate(sub_items):
                local_weights_l2[item] = w_sub[i]
            
            # 시각화
            st.caption(f"{main_cat} 내부 가중치")
            st.dataframe(pd.DataFrame(w_sub, index=sub_items, columns=["가중치"]))
            
        idx += 1
    
    st.session_state['w_l2'] = local_weights_l2

# --- [Tab 3] 3계층 대안 평가 (180일 vs 70일) ---
with tab3:
    st.header("3. 대안 선호도 평가 (180일형 vs 70일형)")
    st.info("각 훈련 항목을 숙달하는 데 있어, 180일형과 70일형 중 언제가 더 유리한가요?")
    
    l3_responses = []
    
    # 5점 척도 직접 가중치 매핑 (3계층은 쌍대비교지만 단순 비율로 처리하는 것이 일반적)
    pref_map = {
        "A 매우 우세(5)": 0.833, "A 우세(3)": 0.75, "동일(1)": 0.5, "B 우세(3)": 0.25, "B 매우 우세(5)": 0.167
    }
    l3_labels = ["A 매우 우세(5)", "A 우세(3)", "동일(1)", "B 우세(3)", "B 매우 우세(5)"]

    for main_cat in l1_items:
        st.subheader(f"📂 {main_cat}")
        for item in hierarchy[main_cat]:
            col_a, col_b = st.columns([2, 3])
            with col_a:
                st.write(f"**{item}**")
            with col_b:
                sel = st.select_slider(
                    f"{item} 비교", options=l3_labels, value="동일(1)", key=f"l3_{item}", label_visibility="collapsed"
                )
                w180 = pref_map[sel]
                w70 = 1.0 - w180
                st.session_state['scores_l3'][item] = (w180, w70)

# --- [Tab 4] 최종 결과 집계 ---
with tab4:
    st.header("🏆 최종 분석 결과")
    
    if st.button("결과 계산 및 표 생성"):
        
        final_rows = []
        total_180_score = 0
        total_70_score = 0
        
        for main_cat in l1_items:
            w1 = st.session_state['w_l1'][main_cat] # 1계층 가중치
            
            for item in hierarchy[main_cat]:
                w2_local = st.session_state['w_l2'][item] # 2계층 지역 가중치
                w_global = w1 * w2_local # 항목의 전체(Global) 중요도
                
                s180, s70 = st.session_state['scores_l3'][item] # 3계층 선호도
                
                # 우세 대안 텍스트
                winner = "180일형" if s180 > s70 else ("70일형" if s70 > s180 else "동일")
                if winner == "70일형": winner = "**70일형**" # 강조
                
                final_rows.append({
                    "대분류": main_cat,
                    "대분류 W": f"{w1:.3f}",
                    "하위 항목": item,
                    "항목 중요도(Global)": w_global,
                    "180일형 선호도": s180,
                    "70일형 선호도": s70,
                    "180일형 점수": w_global * s180,
                    "70일형 점수": w_global * s70,
                    "개별 우세": winner
                })
                
                total_180_score += w_global * s180
                total_70_score += w_global * s70

        df_final = pd.DataFrame(final_rows)
        
        # 보기 좋게 포맷팅
        st.subheader("1. 항목별 상세 분석표")
        st.dataframe(
            df_final[[
                "대분류", "하위 항목", "항목 중요도(Global)", 
                "180일형 선호도", "70일형 선호도", "개별 우세"
            ]].style.format({
                "항목 중요도(Global)": "{:.3f}",
                "180일형 선호도": "{:.3f}",
                "70일형 선호도": "{:.3f}"
            })
        )
        
        st.divider()
        
        st.subheader("2. 최종 종합 평가")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric(label="180일형 총점", value=f"{total_180_score:.4f}")
            st.metric(label="70일형 총점", value=f"{total_70_score:.4f}", 
                      delta=f"{total_70_score - total_180_score:.4f}")
        
        with col_res2:
            final_winner = "180일형" if total_180_score > total_70_score else "70일형"
            st.success(f"최종적으로 **[{final_winner}]**이 더 우수한 대안으로 도출되었습니다.")
            
            # 파이차트
            chart_data = pd.DataFrame({
                "Score": [total_180_score, total_70_score],
                "Alternative": ["180 Days", "70 Days"]
            })
            st.bar_chart(chart_data.set_index("Alternative"))

        # CSV 다운로드
        st.download_button(
            "결과 엑셀(CSV) 다운로드",
            df_final.to_csv(index=False).encode('utf-8-sig'),
            "ahp_final_result.csv"
        )
