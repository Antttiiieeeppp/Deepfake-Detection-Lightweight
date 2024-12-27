import streamlit as st
import pandas as pd
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Anttiep's Dashboard",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.write("# Welcome to Antttiiieeeppp! 👋")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    :rainbow[딥페이크 탐지 모델]은 이미 높은 성능을 갖추고 있으나, 연산 자원이 제한된 환경에서는 활용이 제한적입니다.
    그렇기에 경량화된 딥페이크 탐지 모델로 이러한 기술적 장벽을 낮춰,
    다양한 기업 시스템 및 기존 SNS 플랫폼에 딥페이크 탐지 기능을 손쉽게 통합하고자 합니다.
    딥페이크 확산 방지를 위한 실시간 대응이 가능한 경량화 AI 모델 개발을 목표로 두고 있습니다.
    
    **👈 Select a demo from the sidebar** to check our project!
    ### Want to learn more about us?
    - Check out repository : [👾 Antttiiieeeppp](https://github.com/Antttiiieeeppp/Deepfake-Detection-Lightweight)
    
    - Jump into our : [README](https://github.com/Antttiiieeeppp/Deepfake-Detection-Lightweight/blob/main/README.md)
    
    
"""
)

st.markdown("""
            ### Contributors ✨""")
# Contributor data
contributors = [
    {
        "name": "HyeongJun Kim",
        "github": "https://github.com/hoooddy",
        "avatar": "https://avatars.githubusercontent.com/u/35017649?v=4",
    },
    {
        "name": "HyeonWoo Choi",
        "github": "https://github.com/po2955",
        "avatar": "https://avatars.githubusercontent.com/u/84663334?v=4",
    },
    {
        "name": "JaeHee Lee",
        "github": "https://github.com/JaeHeeLE",
        "avatar": "https://avatars.githubusercontent.com/u/153152453?v=4",
    },
    {
        "name": "Seunga Kim",
        "github": "https://github.com/sinya3558",
        "avatar": "https://avatars.githubusercontent.com/u/70243358?v=4",
    }
]

# Display contributors
cols = st.columns(len(contributors))
for col, contributor in zip(cols, contributors):
    with col:
        st.image(contributor["avatar"], width=100)
        st.markdown(f"**[{contributor['name']}]({contributor['github']})**")