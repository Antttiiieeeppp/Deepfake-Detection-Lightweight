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

# Sidebar for navigation
with st.sidebar:
    st.title("Page Overview")
    page = st.radio("Go to a page:", ["Home", "Upload Video", "About"])

# 1. Home Page
if page == "Home":
    st.title("🎭 Team ANTTTIIIEEEPPP's Demo")
    
    # Project Introduction
    st.header("Project 소개 :sunglasses:", divider=True)
    multi = '''
    :rainbow[딥페이크 탐지 모델]은 이미 높은 성능을 갖추고 있으나, 연산 자원이 제한된 환경에서는 활용이 제한적입니다.
    그렇기에 경량화된 딥페이크 탐지 모델로 이러한 기술적 장벽을 낮춰,
    다양한 기업 시스템 및 기존 SNS 플랫폼에 딥페이크 탐지 기능을 손쉽게 통합하고자 합니다.
    딥페이크 확산 방지를 위한 실시간 대응이 가능한 경량화 AI 모델 개발을 목표로 두고 있습니다.
    '''
    st.markdown(multi)
    
    # Example Video Section
    st.header("프로젝트 배포 예시", divider=True)
    VIDEO_URL = "https://www.youtube.com/watch?v=HhMPSuZgJsc"
    st.video(VIDEO_URL)


# 2. Video Upload Page
elif page == "Upload Video":
    st.title("Detect Deepfakes in Videos")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4"])   # 추가 할것 : avi, mov
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        st.markdown("**Analyzing video for deepfakes...**")
        # Call your detection model here
        # result = detect_video(uploaded_video)  # Example function
        st.success("Deepfake detected!")  # Replace with your actual result


# 3. About Page
elif page == "About":
    st.title("About the Project")
    st.markdown("""
    This project aims to combat the misuse of deepfake technology by providing lightweight AI models 
    capable of real-time detection in resource-limited environments. 
    
    (need to add more here..)
    """)

 # Contributors Section
    st.subheader("Contributors ✨", divider="grey")
    
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
            
# GitHub Repo Section
    st.subheader("GitHub Repository 👾", divider=True)
    repo_text = '''
    If you want to check our further updates, 
    
    👉 **Visit our GitHub Repo:** [💻 Antttiiieeeppp](https://github.com/Antttiiieeeppp/Deepfake-Detection-Lightweight)
    '''
    st.markdown(repo_text)
    
# Location Section
    st.subheader("Where we are 🦝", divider=True)
    st.caption("Toegye-ro, Jung-go,")
    st.caption("Seoul, Republic of Korea")
    