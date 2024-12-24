import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
# hw
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
import yaml
# jh
from torchvision import transforms
# 모델 class 정의
import sys
import os

# 상대 경로를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../efficient-vit")))
# EfficientViT 클래스 임포트
from efficient_vit import EfficientViT
from cnn import SimpleCNN

# from ..efficient_vit import EfficientViT
# sa
from modify_video import add_badge_with_timestamps
BADGE_PATH = "Faces Resource.png"

# efficient_vit_path = "../efficient-vit/efficient_vit.py"
# if efficient_vit_path not in sys.path:
#     sys.path.append(efficient_vit_path)

# functions
# 크롭된 이미지를 모델에 넣기 위한 전처리 함수
def preprocess_image(cropped_face):
    # OpenCV를 사용하여 이미지를 텐서로 변환
    cropped_face = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    cropped_face = cropped_face.resize((224, 224))  # 모델에 맞는 크기
    cropped_face = np.array(cropped_face).transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
    cropped_face = torch.tensor(cropped_face, dtype=torch.float32) / 255.0  # 0~1로 정규화
    cropped_face = cropped_face.unsqueeze(0)  # 배치 차원 추가
    return cropped_face

# 예측 함수
def predict(cropped_face):
    # 이미지 전처리
    input_tensor = preprocess_image(cropped_face)
    
    # 모델에 입력
    with torch.no_grad():
        output = model(input_tensor)  # 모델의 출력
        # 출력이 확률일 경우 (sigmoid를 통과시켜서 0~1로 만들기)
        prob = torch.sigmoid(output)
        
        # 확률을 0.5 기준으로 1 또는 0으로 변환
        prediction = 1 if prob > 0.5 else 0
    return prediction, prob.item()

# 2FPS로 프레임 추출
# def extract_frames(video_path, fps=2):    # 동일한 함수 2개 -> 하나는 주석 처리 하겠습니다
#     cap = cv2.VideoCapture(video_path)
#     original_fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_interval = original_fps // fps
#     frames = []
#     count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if count % frame_interval == 0:
#             frames.append(frame)
#         count += 1

#     cap.release()
#     return frames

# # 얼굴 감지 및 크롭
# def detect_and_crop_faces(frames):
#     cropped_faces = []
#     mp_face_detection = mp.solutions.face_detection
#     face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)  # min_detection_confidence 조정

#     for frame in frames:
#         # 얼굴 감지
#         results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         if results.detections:
#             for detection in results.detections:
#                 bboxC = detection.location_data.relative_bounding_box
#                 h, w, _ = frame.shape
#                 x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                
#                 # 얼굴 크롭
#                 cropped_face = frame[y:y + h_box, x:x + w_box]
                
#                 # 224x224로 리사이즈
#                 resized_face = cv2.resize(cropped_face, (224, 224))
#                 cropped_faces.append(resized_face)
#         else:
#             cropped_faces.append(None)
    
#     return cropped_faces

# Helper functions for video processing
def extract_frames(video_path, fps=2):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    
    # FPS에 맞춰 프레임을 추출
    total_frames = len(frames)
    interval = int(total_frames / (fps * (total_frames / 30)))  # FPS에 맞는 간격 계산
    selected_frames = frames[::interval]  # 2FPS로 프레임 선택

    return selected_frames

def detect_and_crop_faces(frames):
    cropped_faces = []
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    for frame in frames:
        # 얼굴 감지
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                
                # 얼굴 크롭
                cropped_face = frame[y:y + h_box, x:x + w_box]
                
                # 224x224로 리사이즈
                resized_face = cv2.resize(cropped_face, (224, 224))
                cropped_faces.append(resized_face)
    return cropped_faces
###########################
# jh

# 모델 로드 함수

def load_model(model_path: str, device):
    # model = EfficientViT(config=config, channels=1280, selected_efficient_net = 0)
    model = SimpleCNN()
    
    # CPU에서 가중치 로드 (GPU 사용 시 map_location='cuda' 등 변경)
    # model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model



# 이미지 전처리 (face_np -> 텐서)
def preprocess_image(face_np: np.ndarray):
    """
    - face_np: (H, W, 3) NumPy, OpenCV(BGR) 형식의 얼굴 이미지.
    - (1) OpenCV -> PIL(RGB) 변환
    - (2) ToTensor + Resize 등으로 모델 입력 형태로 변환
    """
    transform = transforms.Compose([
        transforms.ToTensor(),             
        transforms.Resize((224, 224)),     
    ])
    
    # OpenCV(BGR) -> PIL(RGB)
    face_pil = Image.fromarray(cv2.cvtColor(face_np, cv2.COLOR_BGR2RGB))
    tensor = transform(face_pil).unsqueeze(0)  #
    return tensor

def detect_deepfake(model, cropped_faces_dict):
    results = {}

    for idx, face_np in cropped_faces_dict.items():
        if face_np is None:
            # 얼굴이 없으면 예측 불가 -> None 처리
            results[idx] = None
            continue

        # 전처리 (preprocess_image 함수 사용)
        input_tensor = preprocess_image(face_np)  # (1, C, H, W) 텐서
# <<<<<<< 이부분?
        input_tensor = input_tensor.to(device)
# =======


        # 모델 추론
        with torch.no_grad():
            output = model(input_tensor)         # shape: (1,1) 가정
            prob = torch.sigmoid(output).item()  # 0~1 확률
            prediction = 1 if prob > 0.7 else 0   # threshold=0.5

        # 결과: {인덱스: 예측값} 형태로 저장
        results[idx] = prediction

    return results
##########################
# real .yaml file
def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


device = "cuda" if torch.cuda.is_available() else "cpu"
# Call your detection model here
CONFIG_PATH = "../efficient-vit/configs/architecture.yaml"  # Replace with the actual path
config = load_yaml_config(CONFIG_PATH)
# 모델 로드
# MODEL_PATH = "../efficient-vit/pretrained_models/efficient_vit.pth"
MODEL_PATH = "../efficient-vit/pretrained_models/kd_cnn.pth"
model = load_model(MODEL_PATH, device)
###############################

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
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_video.read())
            video_path = temp_video.name
            # output_video_path = temp_video.name ####
        
        # 2FPS로 프레임 추출

        frames = extract_frames(video_path, fps=2)
        total_frames = len(frames)
        st.write(f"Extracted {total_frames} frames from the video.")
        
        # 얼굴 감지 및 크롭
        cropped_faces = detect_and_crop_faces(frames)
        total_cropped_faces = len(cropped_faces)
        st.write(f"Detected and cropped {total_cropped_faces} faces.")
        # st.write(cropped_faces)
        # # 크롭된 얼굴 출력
        # if cropped_faces:
        #     st.write("Cropped Faces:")
        #     for i, face in enumerate(cropped_faces):
        #         face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        #         st.image(face_image, caption=f"Cropped Face {i+1}", width=150)  # width를 150으로 설정
        # else:
        #     st.write("No faces detected in the video.")
        # 임시 파일 삭제
        # os.remove(video_path)
        # 크롭된 이미지 딕셔너리로 만들기
        cropped_faces_dict = {index: value for index, value in enumerate(cropped_faces)}
        #만든 딕셔터리 출력
        #st.write(cropped_faces_dict)
        # #
        # # Call your detection model here
        # # 모델 로드
        # MODEL_PATH = "/home/work/Antttiiieeeppp/jh-Lightweight/pretrained_models/effcient_vit.pth"
        # model = load_model(MODEL_PATH)

        # # cropped_faces_dict = {
        # #     0: face_np_0,  # 실제 얼굴 이미지(np.ndarray)
        # #     1: None,       # 얼굴이 없는 경우 예시
        # #     2: face_np_2
        # # }

        # # 딥페이크 판별
        predictions_dict = detect_deepfake(model, cropped_faces_dict)
        # 테스트
        print(predictions_dict)

        if isinstance(predictions_dict, dict):  # Check if predictions_dict is a valid dictionary
            try:
                # Add badge to the video at specific timestamps
                modified_video = add_badge_with_timestamps(video_path, BADGE_PATH, predictions_dict)
                
                # Display the processed video in the Streamlit app
                st.video(modified_video)
                st.success("Deepfake detected!")
            except Exception as e:
                st.error(f"An error occurred while processing the video: {e}")
        else:
            st.warning("Could not analyze the video. Please try again.")
        # if isinstance(predictions_dict, dict):  # If tis a dictionary -> {1: 0, 2: 1, 3: 1}
        #     # Add subtitles at specific timestamps
        #     # modified_video = adding_subtitles_to_video_w_dict(video_path, predictions_dict) 
        #     modified_video = add_badge_with_timestamps(video_path, BADGE_PATH, output_video_path, predictions_dict)
        #     st.video(modified_video)   
        #     st.success("Deepfake detected!")
        # else:
        #     st.warning("Could not analyze the video. Please try again.")
        # st.success("된다!")
        
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
    