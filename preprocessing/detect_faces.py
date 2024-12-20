import argparse
import json
import os
import numpy as np
from typing import Type
from datetime import datetime

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import face_detector
from face_detector import VideoDataset
from face_detector import VideoFaceDetector
from utils import get_video_paths, get_method
import argparse

import time ## [추가] 추론시간측정

import torch
# [추가] CPU 설정
torch.set_num_threads(1)  # CPU 스레드 수 제한
device = "cpu"  # 강제로 CPU 사용

# [추가] 현재 시간으로 디렉토리 이름 생성
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULT_PATH = f"/home/work/Antttiiieeeppp/jh-Lightweight/jaehee/result/{current_time}"
os.makedirs(RESULT_PATH, exist_ok=True)

def process_videos(videos, detector_cls: Type[VideoFaceDetector], selected_dataset, opt):
    #detector = face_detector.__dict__[detector_cls](device="cuda:0")
    detector = face_detector.__dict__[detector_cls](device=device)

    dataset = VideoDataset(videos)
    
    loader = DataLoader(dataset, shuffle=False, num_workers=opt.processes, batch_size=1, collate_fn=lambda x: x)
    
    missed_videos = []
    total_time = 0  # [추가] 총 추론 시간 저장 변수
    total_frames = 0  # [추가] 총 처리된 프레임 수

    for item in tqdm(loader): 
        result = {}
        video, indices, frames = item[0]
        if selected_dataset == 1:
            method = get_method(video, opt.data_path)
            out_dir = os.path.join(opt.data_path, "boxes", method)
        else:
            out_dir = os.path.join(opt.data_path, "boxes")

        id = os.path.splitext(os.path.basename(video))[0]

        if os.path.exists(out_dir) and "{}.json".format(id) in os.listdir(out_dir):
            continue
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
      
        for j, frames in enumerate(batches):
          start_time = time.time()  # [추가] 추론 시작 시간
          batch_result = detector._detect_faces(frames)
          end_time = time.time()  # [추가] 추론 종료 시간
          
          result.update({int(j * detector._batch_size) + i : b for i, b in zip(indices, detector._detect_faces(frames))})

          # [추가] 추론 시간 및 프레임 수 집계
          total_time += (end_time - start_time)
          total_frames += len(frames)
       
        os.makedirs(out_dir, exist_ok=True)
        print(len(result))
        if len(result) > 0:
            with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
                json.dump(result, f)
        else:
            missed_videos.append(id)
        
    avg_inference_time = total_time / total_frames if total_frames > 0 else 0
    # [추가] 추론 시간 기록
    experiment_content = f'''Preprocessing step: detect_faces
    Total frames: {total_frames}
    Total time: {total_time:.2f}s
    Avg inference time: {avg_inference_time:.6f} s/frame
    '''

    with open(os.path.join(RESULT_PATH, 'experiment_content_detect.txt'), 'w', encoding='utf-8') as file:
        file.write(experiment_content)

    return total_time, total_frames  # [추가] 추론 시간과 처리된 프레임 수 반환

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DFDC", type=str,
                        help='Dataset (DFDC / FACEFORENSICS)')
    parser.add_argument('--data_path', default='', type=str,
                        help='Videos directory')
    parser.add_argument("--detector-type", help="Type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    parser.add_argument("--processes", help="Number of processes", default=1)
    parser.add_argument("--fraction", help="Fraction of dataset to use (e.g., 0.1 for 10%)", type=float, default=0.1)  # [추가] 사용할 데이터셋 비율
    opt = parser.parse_args()
    print(opt)


    if opt.dataset.upper() == "DFDC":
        dataset = 0
    else:
        dataset = 1

    videos_paths = []
    if dataset == 1:
        videos_paths = get_video_paths(opt.data_path, dataset)
    else:
        os.makedirs(os.path.join(opt.data_path, "boxes"), exist_ok=True)
        already_extracted = os.listdir(os.path.join(opt.data_path, "boxes"))
        for folder in os.listdir(opt.data_path):
            if "boxes" not in folder and "zip" not in folder:
                if os.path.isdir(os.path.join(opt.data_path, folder)): # For training and test set
                    for video_name in os.listdir(os.path.join(opt.data_path, folder)):
                        if video_name.split(".")[0] + ".json" in already_extracted:
                            continue
                        videos_paths.append(os.path.join(opt.data_path, folder, video_name))
                else: # For validation set
                    videos_paths.append(os.path.join(opt.data_path, folder))

    # [추가] 데이터셋 비율로 리스트를 줄이기
    total_videos = len(videos_paths)
    subset_size = int(total_videos * opt.fraction)
    videos_paths = videos_paths[:subset_size]  # 처음부터 fraction에 해당하는 부분만 사용
    print(f"Using {subset_size} out of {total_videos} videos ({opt.fraction * 100:.0f}% of dataset)")

    process_videos(videos_paths, opt.detector_type, dataset, opt)


if __name__ == "__main__":
    main()