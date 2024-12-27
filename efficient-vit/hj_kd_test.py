import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
import os, platform
import cv2
import numpy as np
import torch
from torch import nn, einsum

from sklearn.metrics import f1_score, roc_curve, auc
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize

from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from efficient_vit import EfficientViT
from utils import transform_frame
import glob
from os import cpu_count
import json
from multiprocessing.pool import Pool
from progress.bar import Bar
import pandas as pd
from tqdm import tqdm
from multiprocessing import Manager
from utils import custom_round, custom_video_round
from deepfakes_dataset import DeepFakesDataset
from torchvision import models, transforms
import time
from sklearn.metrics import precision_score, recall_score
import GPUtil

import yaml
import argparse
import gc

import platform
import psutil
import cpuinfo
import distro

torch.cuda.empty_cache()
gc.collect()

device = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "result/2024-12-17_16-27-54"
BASE_DIR = "../../deep_fakes"

DATA_DIR = os.path.join(BASE_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
RESULT_PATH = os.path.join(MODELS_DIR, "test_result")
MODEL_PATH = os.path.join(MODELS_DIR, "efficientnetB0")

# 디렉토리 생성
os.makedirs(RESULT_PATH, exist_ok=True)

TEST_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")

def create_base_transform(size):
    return Compose([
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
    ])

def plot_roc_auc(val_labels_list, val_preds):
    # ROC 곡선 계산
    fpr, tpr, thresholds = roc_curve(val_labels_list, val_preds)
    roc_auc = auc(fpr, tpr)  # AUC 값 계산

    # ROC 곡선 그리기
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 무작위 예측선
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    # 그래프 저장
    plt.savefig(os.path.join(RESULT_PATH, 'roc_auc_curve.png'))
    return roc_auc

def read_frames(video_path, videos):
    
    test_df = pd.DataFrame(pd.read_csv(TEST_LABELS_PATH))
    video_folder_name = os.path.basename(video_path)
    video_key = video_folder_name + ".mp4"
    label = test_df.loc[test_df['filename'] == video_key]['label'].values[0]

    for frame_path in glob.glob(os.path.join(video_path, "*")):
        image = cv2.imread(frame_path)
        if image is not None:
            image = cv2.resize(image, (config['model']['image-size'], config['model']['image-size']))
            videos.append((image, label))


def get_gpu_info():
    # GPU 정보 가져오기
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "ID": gpu.id,
            "Name": gpu.name,  # GPU 모델명
            "Memory Total": gpu.memoryTotal,
            "Memory Free": gpu.memoryFree,
            "Memory Used": gpu.memoryUsed,
            "GPU Utilization": gpu.load * 100,  # %로 변환
            "GPU Memory Utilization": gpu.memoryUtil * 100,  # %로 변환
            "Temperature": gpu.temperature
        })
    return gpu_info

# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                        help='Path to model checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='DFDC', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|DFDC)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--frames_per_video', type=int, default=30, 
                        help="How many equidistant frames for each video (default: 30)")
    parser.add_argument('--batch_size', type=int, default=32, 
                        help="Batch size (default: 32)")
    
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    if opt.efficient_net == 0:
        channels = 1280
    else:
        channels = 2560

    if os.path.exists(MODEL_PATH):
        # model = EfficientViT(config=config, channels=channels, selected_efficient_net = opt.efficient_net)
        model = models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, 1),
    )
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        model = model.cuda()
    else:
        print("No model found.")
        exit()

    print(get_n_params(model))

    model_name = os.path.basename(MODEL_PATH)

    
    preds = []
    paths = []
    videos = []

    folders = ["DFDC"]

    for folder in folders:
        method_folder = os.path.join(TEST_DIR, folder)  
        for index, video_folder in enumerate(os.listdir(method_folder)):
            paths.append(os.path.join(method_folder, video_folder))
      

    for path in tqdm(paths):
        read_frames(path, videos=videos)

    correct_test_labels = np.asarray([row[1] for row in videos])
    videos = np.asarray([row[0] for row in videos])
    print(videos.shape)


    test_dataset = DeepFakesDataset(videos, correct_test_labels, config['model']['image-size'], mode='validation')
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    
    del test_dataset

    preds = []

    bar = Bar('Predicting', max=len(test_dl))

    test_correct = 0
    test_positive = 0
    test_negative = 0
    test_counter = 0

    test_preds = []  # 검증 예측값 저장 리스트
    test_prob_list = []  # 검증 데이터에 대한 예측 확률 리스트
    test_labels_list = []  # 검증 실제 레이블 저장 리스트
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total_test_loss = 0
    inference_time_list = []
    for index, (test_images, test_labels) in enumerate(test_dl):

        test_images = np.transpose(test_images, (0, 3, 1, 2))
        
        # test_images = test_images.cuda()
        test_images = test_images.to(device)
        test_labels = test_labels.unsqueeze(1).float()

        inference_start = time.time()
        test_pred = model(test_images)
        inference_time_list.append(round(time.time() - inference_start, 4))

        test_pred = test_pred.cpu().float()
        
        test_loss = loss_fn(test_pred, test_labels)
        total_test_loss += round(test_loss.item(), 2)
        corrects, positive_class, negative_class = check_correct(test_pred, test_labels)
        test_correct += corrects
        test_positive += positive_class
        test_counter += 1
        test_negative += negative_class

        test_prob_list.extend(torch.sigmoid(test_pred).detach().numpy().flatten())
        
        test_preds.extend(torch.sigmoid(test_pred).detach().numpy().round())  # 예측값 추가            
        test_labels_list.extend(test_labels.cpu().numpy())  # 실제 레이블 추가

        bar.next()
    
    bar.finish()

    test_f1_score = round(f1_score(test_labels_list, test_preds, zero_division=1), 4)  # 검증 F1 score 계산
    precision = precision_score(test_labels_list, test_preds)
    recall = recall_score(test_labels_list, test_preds)
    
    auc = plot_roc_auc(test_labels_list, test_prob_list)

    gpu_info = get_gpu_info()
    cpu_info = cpuinfo.get_cpu_info()
    
    # 파일에 쓸 내용
    experiment_content = f'''OS: {platform.system()}
Version: {distro.name()} {distro.version()}
Processor: {platform.processor()}
Architecture: {platform.architecture()}

CPU: {cpu_info['brand_raw']}
CPU_ARCH: {cpu_info['arch']}
CPU_CORE: {cpu_info['count']}

GPU: {gpu_info[0]["Name"]}
GPU Memory: {gpu_info[0]["Memory Total"]}

Model_Name: EfficientNetB0
Model_Parameter:{get_n_params(model)}
Device: {device}
Precision: {round(precision, 4)}
Recall: {round(recall, 4)}
F1_score: {round(test_f1_score, 4)}
AUC: {round(auc, 4)}
AVG Inference Time: {sum(inference_time_list) / len(inference_time_list)}

Note: Knowledge Distillation
    Teacher: Efficient-ViT
    Student: EfficientNetB0
'''
    print(experiment_content)
#     =-=-=-=-=-=-=-=
    # 파일을 쓰기 모드로 열기 (파일이 없으면 새로 생성됨)
    with open(os.path.join(RESULT_PATH, 'experiment_content.txt'), 'w', encoding='utf-8') as file:
        file.write(experiment_content)



