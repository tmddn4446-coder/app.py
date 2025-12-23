import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# [설정] 페이지 기본 설정
# ------------------------------------------------------------------
st.set_page_config(page_title="장기 상비예비군 훈련 우선순위 조사", layout="wide")

# 한글 폰트 설정 (스트림릿 클라우드 환경 호환용)
import matplotlib.font_manager as fm
import os

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

# 기본적으로 sans-serif 설정 (한글 깨짐 방지 노력)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------------
# [함수] AHP 계산 엔진
# ------------------------------------------------------------------
def calculate_ahp(matrix):
    n = len(matrix)
    col_sums = matrix.sum(axis=0)
    norm_matrix = matrix / col_sums
    weights = norm_matrix.mean(axis=1)
    
    # 일관성 비율(CR) 계산
    weighted_sum = np.dot(matrix, weights)
    lambda_max = np.mean(weighted_sum / weights)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12}
    ri = ri_dict.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    
    return weights, cr

# ------------------------------------------------------------------
# [함수] 5점 척도 UI 생성기
# ------------------------------------------------------------------
def ahp_question(label, item_a, item_b, key_suffix):
    st.markdown(f"**[{label}]**")
    val = st.select_slider(
        f"'{item_a}' vs '{item_b}' 중요도 비교",
        options=[-7, -3, 1, 3, 7],
        value=1,
        format_func=lambda x: 
            f"{item_a} 매우 중요(7)" if x == 7 else
            f"{item_a} 중요(3)" if x == 3 else
            "동등(1)" if x == 1 else
            f"{item_b} 중요(3)" if x == -3 else
            f"{item_b} 매우 중요(7)",
        key=f"q_{key_suffix}"
    )
    # 선택값을 AHP 수치로 변환
    if val == 1: return 1.0
    elif val > 0: return float(val) # A가 중요
    else: return 1.0 / abs(val)     # B가 중요 (역수)

# ------------------------------------------------------------------
# [메인] 웹 앱 레이아웃
# ------------------------------------------------------------------
st.title("🎖️ 장기 상비예비군 훈련 프로그램 우선순위 설문")
st.markdown("""
이 설문은 **장기 상비예비군(180일/70일)**의 훈련 효과성을 분석하기 위한 AHP 조사입니다.  
각 항목 간의 상대적 중요도를 선택해 주세요.
""")

with st.sidebar:
    st.header("응답자 정보")
    role = st.selectbox("직책을 선택하세요", ["지휘관", "상비예비군", "정책담당자", "기타"])
    st.info("모든 문항에 응답 후 하단의 '결과 분석' 버튼을 눌러주세요.")

# --- 1. 제1계층 평가 ---
st.header("1. 평가 기준 (제1계층)")
with st.expander("평가 기준 중요도 비교 (클릭하여 열기)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        a12 = ahp_question("비교 1", "전투기술", "작계시행", "l1_1")
        a13 = ahp_question("비교 2", "전투기술", "교관능력", "l1_2")
    with col2:
        a23 = ahp_question("비교 3", "작계시행", "교관능력", "l1_3")

# --- 2. 제2계층 평가 ---
st.header("2. 세부 훈련 과목 평가 (제2계층)")

tab1, tab2, tab3 = st.tabs(["전투기술 하위", "작계시행 하위", "교관능력 하위"])

with tab1:
    c12 = ahp_question("전투기술", "전시물자 지원", "개인화기 사격", "c_1")
    c13 = ahp_question("전투기술", "전시물자 지원", "편제장비 운용", "c_2")
    c23 = ahp_question("전투기술", "개인화기 사격", "편제장비 운용", "c_3")

with tab2:
    o12 = ahp_question("작계시행", "지형정찰", "증창설 절차", "o_1")
    o13 = ahp_question("작계시행", "지형정찰", "지휘통제기구", "o_2")
    o23 = ahp_question("작계시행", "증창설 절차", "지휘통제기구", "o_3")

with tab3:
    i12 = ahp_question("교관능력", "교관 자격인증", "병 진급평가", "i_1")
    i13 = ahp_question("교관능력", "교관 자격인증", "공용화기 교육", "i_2")
    i23 = ahp_question("교관능력", "병 진급평가", "공용화기 교육", "i_3")

# --- 3. 대안 평가 (간소화) ---
st.header("3. 제도별 효율성 평가")
st.markdown("각 훈련 과목에 대해 **180일형**과 **70일형** 중 어느 쪽이 효율적인지 선택하세요.")
# 9개 과목에 대한 180일형 선호도 입력
alt_scores = {}
items = [
    "전시물자 지원", "개인화기 사격", "편제장비 운용", 
    "지형정찰", "증창설 절차", "지휘통제기구",
    "교관 자격인증", "병 진급평가", "공용화기 교육"
]

cols = st.columns(3)
for i, item in enumerate(items):
    with cols[i%3]:
        val = st.select_slider(
            f"**{item}**",
            options=["70일형 유리", "비슷", "180일형 유리"],
            value="180일형 유리" if item != "병 진급평가" else "70일형 유리", # 기본값 세팅
            key=f"alt_{i}"
        )
        # 가중치 매핑 (약식)
        if val == "180일형 유리": alt_scores[item] = 0.75
        elif val == "비슷": alt_scores[item] = 0.5
        else: alt_scores[item] = 0.25

# --- 결과 분석 버튼 ---
if st.button("📊 결과 분석 및 리포트 생성", type="primary"):
    
    # 1. 행렬 생성 및 계산
    m_l1 = np.array([[1, a12, a13], [1/a12, 1, a23], [1/a13, 1/a23, 1]])
    m_c = np.array([[1, c12, c13], [1/c12, 1, c23], [1/c13, 1/c23, 1]])
    m_o = np.array([[1, o12, o13], [1/o12, 1, o23], [1/o13, 1/o23, 1]])
    m_i = np.array([[1, i12, i13], [1/i12, 1, i23], [1/i13, 1/i23, 1]])
    
    w1, cr1 = calculate_ahp(m_l1)
    w_c, cr_c = calculate_ahp(m_c)
    w_o, cr_o = calculate_ahp(m_o)
    w_i, cr_i = calculate_ahp(m_i)
    
    # CR 검증 알림
    max_cr = max(cr1, cr_c, cr_o, cr_i)
    if max_cr > 0.1:
        st.warning(f"⚠️ 일부 응답의 일관성 비율(CR)이 0.1을 초과했습니다 (최대 CR: {max_cr:.3f}). 신중한 재응답을 권장합니다.")
    else:
        st.success(f"✅ 모든 응답의 논리적 일관성이 확보되었습니다 (최대 CR: {max_cr:.3f}).")
    
    # 2. 종합 가중치 계산
    global_w = []
    global_w.extend(w1[0] * w_c)
    global_w.extend(w1[1] * w_o)
    global_w.extend(w1[2] * w_i)
    
    # 3. 대안 점수 계산
    score_180 = 0
    score_70 = 0
    
    for idx, item in enumerate(items):
        w_180_item = alt_scores[item]
        w_70_item = 1 - w_180_item
        score_180 += global_w[idx] * w_180_item
        score_70 += global_w[idx] * w_70_item
        
    # 4. 결과 시각화
    st.divider()
    st.subheader(f"🏆 분석 결과 ({role} 관점)")
    
    res_df = pd.DataFrame({
        "대안": ["180일형", "70일형"],
        "효과성 점수": [score_180, score_70]
    })
    
    # B/C 분석 추가 (기간을 비용으로 가정)
    cost_180 = 2.57 # 180일 / 70일
    res_df["상대적 비용"] = [cost_180, 1.0]
    res_df["ROI (가성비)"] = res_df["효과성 점수"] / res_df["상대적 비용"]
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown("### 1. 절대적 효과성 (Effectiveness)")
        st.bar_chart(res_df.set_index("대안")["효과성 점수"], color="#2E8B57")
        st.info(f"전투력 기여도: 180일형({score_180:.3f}) vs 70일형({score_70:.3f})")
        
    with col_res2:
        st.markdown("### 2. 비용 대비 효율 (ROI)")
        st.bar_chart(res_df.set_index("대안")["ROI (가성비)"], color="#FF6347")
        roi_180 = res_df.loc[0, "ROI (가성비)"]
        roi_70 = res_df.loc[1, "ROI (가성비)"]
        
        if roi_180 > roi_70:
            st.success(f"비용(2.57배)을 고려해도 180일형의 효율이 더 높습니다.")
        else:
            st.warning(f"비용을 고려하면 70일형의 효율이 더 높습니다.")
    
    # 상세 테이블
    st.write("### 📋 세부 항목별 중요도")
    detail_df = pd.DataFrame({
        "훈련 과목": items,
        "종합 중요도": global_w,
        "180일형 선호도": [alt_scores[i] for i in items]
    }).sort_values("종합 중요도", ascending=False)
    
    st.dataframe(detail_df.style.background_gradient(cmap="Blues"), use_container_width=True)
