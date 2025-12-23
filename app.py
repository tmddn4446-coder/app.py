import streamlit as st
import pandas as pd
import numpy as np

# 페이지 설정
st.set_page_config(page_title="AHP 통합 분석 시스템", layout="wide")

st.title("🪖 교육훈련 기간 체계 개선 AHP 설문 (일관성 검증 포함)")
st.markdown("""
이 시스템은 **[1계층] → [2계층] → [3계층]** 순서로 쌍대비교를 수행합니다.  
⚠️ **경고 기능:** 응답의 논리적 모순(일관성 비율 > 0.1)이 발견되면 경고 메시지가 표시됩니다.
""")

# ---------------------------------------------------------
# [핵심 함수] AHP 계산 및 일관성 검증 (CR 계산)
# ---------------------------------------------------------
# Random Index (RI) - 행렬 크기(n)별 상수 값 (n=1~10)
RI_DICT = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}

def calculate_ahp_with_consistency(matrix):
    """
    고유치법(Eigenvalue method)을 사용하여 가중치와 CR(Consistency Ratio)을 계산합니다.
    """
    n = matrix.shape[0]
    
    # 1. 고유값/고유벡터 계산
    eig_vals, eig_vecs = np.linalg.eig(matrix)
    max_eig_val = np.max(eig_vals).real # 최대 고유값 (Lambda Max)
    eig_vec = eig_vecs[:, np.argmax(eig_vals)].real
    
    # 2. 가중치 정규화 (합이 1이 되도록)
    weights = eig_vec / np.sum(eig_vec)
    
    # 3. 일관성 지수 (CI) 계산
    ci = (max_eig_val - n) / (n - 1)
    
    # 4. 일관성 비율 (CR) 계산
    if n in RI_DICT and RI_DICT[n] != 0:
        cr = ci / RI_DICT[n]
    else:
        cr = 0.0 # n=1, 2일 때는 항상 일관성 0 (완벽)
        
    return weights, cr

# 5점 척도 설정
scale_labels = ["A 매우 중요(5)", "A 중요(3)", "동일(1)", "B 중요(3)", "B 매우 중요(5)"]
scale_values = {
    "A 매우 중요(5)": 5.0, "A 중요(3)": 3.0, "동일(1)": 1.0, "B 중요(3)": 1/3.0, "B 매우 중요(5)": 1/5.0
}

def pairwise_input(label, item_a, item_b, key):
    st.write(f"**{item_a}** vs **{item_b}**")
    choice = st.select_slider(
        label, options=scale_labels, value="동일(1)", key=key, label_visibility="collapsed"
    )
    return scale_values[choice]

# 데이터 구조
hierarchy = {
    "전투기술": ["전시물자 군수지원", "개인화기 사격", "편제장비 운용"],
    "작계시행": ["지휘통제기구 훈련", "작계지역 지형정찰", "증·창설 절차 숙달"],
    "교관능력": ["전시교관 자격인증평가", "공용화기 집체교육", "병진급개인기본훈련평가"]
}
l1_items = list(hierarchy.keys())

# 세션 상태 초기화
if 'w_l1' not in st.session_state: st.session_state['w_l1'] = {}
if 'w_l2' not in st.session_state: st.session_state['w_l2'] = {}
if 'scores_l3' not in st.session_state: st.session_state['scores_l3'] = {}

# ---------------------------------------------------------
# [Tab 1] 1계층 평가 (일관성 체크 적용)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["1계층: 대분류", "2계층: 하위항목", "3계층: 대안비교", "🏆 최종 결과"])

with tab1:
    st.header("1. 대분류 중요도 평가")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # A vs B / A vs C / B vs C
        comp_1_2 = pairwise_input("1vs2", l1_items[0], l1_items[1], "l1_1")
        comp_1_3 = pairwise_input("1vs3", l1_items[0], l1_items[2], "l1_2")
        comp_2_3 = pairwise_input("2vs3", l1_items[1], l1_items[2], "l1_3")

        # 행렬 구성
        mat_l1 = np.array([
            [1.0, comp_1_2, comp_1_3],
            [1/comp_1_2, 1.0, comp_2_3],
            [1/comp_1_3, 1/comp_2_3, 1.0]
        ])
        
        # 계산
        w_l1, cr_l1 = calculate_ahp_with_consistency(mat_l1)
        st.session_state['w_l1'] = dict(zip(l1_items, w_l1))

    with col2:
        st.subheader("분석 결과")
        
        # 일관성 경고 메시지 로직
        if cr_l1 > 0.1:
            st.error(f"⚠️ **일관성 부족 (CR = {cr_l1:.3f})**")
            st.markdown("""
            논리적 모순이 발생했습니다. (예: A>B, B>C 인데 C>A 선택)  
            **CR 값이 0.1 이하**가 되도록 좌측 응답을 조정해주세요.
            """)
        else:
            st.success(f"✅ **논리적 일관성 확보 (CR = {cr_l1:.3f})**")
            
        st.bar_chart(pd.Series(st.session_state['w_l1']))

# ---------------------------------------------------------
# [Tab 2] 2계층 평가 (일관성 체크 적용)
# ---------------------------------------------------------
with tab2:
    st.header("2. 하위 훈련항목 중요도 평가")
    
    local_weights_l2 = {}
    cols = st.columns(3)
    idx = 0
    
    for main_cat in l1_items:
        sub_items = hierarchy[main_cat]
        
        with cols[idx]:
            st.subheader(f"📌 {main_cat}")
            
            v1 = pairwise_input(f"{main_cat}_1", sub_items[0], sub_items[1], f"l2_{main_cat}_1")
            v2 = pairwise_input(f"{main_cat}_2", sub_items[0], sub_items[2], f"l2_{main_cat}_2")
            v3 = pairwise_input(f"{main_cat}_3", sub_items[1], sub_items[2], f"l2_{main_cat}_3")
            
            mat_sub = np.array([
                [1.0, v1, v2],
                [1/v1, 1.0, v3],
                [1/v2, 1/v3, 1.0]
            ])
            
            w_sub, cr_sub = calculate_ahp_with_consistency(mat_sub)
            
            # CR 체크 및 경고
            if cr_sub > 0.1:
                st.error(f"⚠️ CR={cr_sub:.3f} (수정 필요)")
            else:
                st.success(f"✅ CR={cr_sub:.3f} (양호)")
            
            # 결과 저장 및 표시
            for i, item in enumerate(sub_items):
                local_weights_l2[item] = w_sub[i]
                
            st.dataframe(pd.DataFrame(w_sub, index=sub_items, columns=["가중치"]).style.format("{:.3f}"))
            
        idx += 1
        
    st.session_state['w_l2'] = local_weights_l2

# ---------------------------------------------------------
# [Tab 3] 3계층 (대안 비교는 일관성 체크 불필요 - 단순 선호도)
# ---------------------------------------------------------
with tab3:
    st.header("3. 대안 선호도 평가")
    pref_map = {"A 매우 우세(5)": 0.833, "A 우세(3)": 0.75, "동일(1)": 0.5, "B 우세(3)": 0.25, "B 매우 우세(5)": 0.167}
    l3_labels = list(pref_map.keys())

    for main_cat in l1_items:
        st.markdown(f"**[{main_cat}] 항목별 비교**")
        for item in hierarchy[main_cat]:
            col_a, col_b = st.columns([2, 3])
            with col_a: st.write(f"- {item}")
            with col_b:
                sel = st.select_slider(f"{item}_slider", options=l3_labels, value="동일(1)", key=f"l3_{item}", label_visibility="collapsed")
                w180 = pref_map[sel]
                st.session_state['scores_l3'][item] = (w180, 1.0 - w180)

# ---------------------------------------------------------
# [Tab 4] 결과 (기존 코드와 동일)
# ---------------------------------------------------------
with tab4:
    st.header("🏆 최종 분석 결과")
    if st.button("결과 계산"):
        final_rows = []
        t180 = 0
        t70 = 0
        
        for main_cat in l1_items:
            for item in hierarchy[main_cat]:
                w_global = st.session_state['w_l1'][main_cat] * st.session_state['w_l2'][item]
                s180, s70 = st.session_state['scores_l3'][item]
                winner = "180일형" if s180 > s70 else ("**70일형**" if s70 > s180 else "동일")
                
                final_rows.append({
                    "대분류": main_cat, "하위 항목": item, "중요도": w_global,
                    "180일형": s180, "70일형": s70, "우세": winner
                })
                t180 += w_global * s180
                t70 += w_global * s70

        res_df = pd.DataFrame(final_rows)
        st.dataframe(res_df.style.format({"중요도": "{:.3f}", "180일형": "{:.3f}", "70일형": "{:.3f}"}))
        
        col_f1, col_f2 = st.columns(2)
        with col_f1: st.metric("180일형 총점", f"{t180:.4f}")
        with col_f2: st.metric("70일형 총점", f"{t70:.4f}", delta=f"{t70-t180:.4f}")
        
        st.success(f"최종 승자: **{'180일형' if t180 > t70 else '70일형'}**")
