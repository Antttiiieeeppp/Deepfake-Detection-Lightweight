import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# 모델 class 정의
efficient_vit_path = "/home/work/Antttiiieeeppp/jh-Lightweight/efficient-vit"
if efficient_vit_path not in sys.path:
    sys.path.append(efficient_vit_path)

from efficient_vit import EfficientViT

# 모델 로드 함수
def load_model(model_path: str):
    model = EfficientViT()
    
    # CPU에서 가중치 로드 (GPU 사용 시 map_location='cuda' 등 변경)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
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

        # 모델 추론
        with torch.no_grad():
            output = model(input_tensor)         # shape: (1,1) 가정
            prob = torch.sigmoid(output).item()  # 0~1 확률
            prediction = 1 if prob > 0.5 else 0   # threshold=0.5

        # 결과: {인덱스: 예측값} 형태로 저장
        results[idx] = prediction

    return results

if __name__ == "__main__":
    # 모델 로드
    MODEL_PATH = "/home/work/Antttiiieeeppp/jh-Lightweight/pretrained_models/effcient_vit.pth"
    model = load_model(MODEL_PATH)
    
    cropped_faces_dict = {
        0: face_np_0,  # 실제 얼굴 이미지(np.ndarray)
        1: None,       # 얼굴이 없는 경우 예시
        2: face_np_2
    }

    # 딥페이크 판별
    predictions_dict = detect_deepfake(model, cropped_faces_dict)

    print(predictions_dict)