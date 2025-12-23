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
    else: return 1.0 / abs(val)     # B가
