import argparse
import json
import os
from os import cpu_count
from pathlib import Path
import time ##[추가] 추론시간측정
import torch ## [추가] CPU 설정을 위한 모듈
from datetime import datetime

# [추가] CPU 설정
torch.set_num_threads(1)  # CPU 스레드 수 제한
device = "cpu"  # 강제로 CPU 사용

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm

from utils import get_video_paths, get_method, get_method_from_name

# [추가] 현재 시간으로 디렉토리 이름 생성
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULT_PATH = f"/home/work/Antttiiieeeppp/jh-Lightweight/jaehee/result/{current_time}"
os.makedirs(RESULT_PATH, exist_ok=True)

def extract_video(video, root_dir, dataset):
    try:
        total_crop_time = 0  # [추가] 크롭 처리 시간 저장 변수
        total_crops = 0  # [추가] 처리된 크롭 수

        if dataset == 0:
            bboxes_path = os.path.join(opt.data_path, "boxes", os.path.splitext(os.path.basename(video))[0] + ".json")
        else:
            bboxes_path = os.path.join(opt.data_path, "boxes", get_method_from_name(video), os.path.splitext(os.path.basename(video))[0] + ".json")
        
        if not os.path.exists(bboxes_path) or not os.path.exists(video):
            return
        with open(bboxes_path, "r") as bbox_f:
            bboxes_dict = json.load(bbox_f)

        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        counter = 0
        for i in range(frames_num):
            capture.grab()
            #if i % 2 != 0:
            #    continue
            success, frame = capture.retrieve()
            if not success or str(i) not in bboxes_dict:
                continue
            id = os.path.splitext(os.path.basename(video))[0]
            crops = []
            bboxes = bboxes_dict[str(i)]
            if bboxes is None:
                continue
            else:
                counter += 1

            start_time = time.time() #[추가] 크롭 시작 시간
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                w = xmax - xmin
                h = ymax - ymin
                p_h = 0
                p_w = 0
                
                #p_h = h // 3
                #p_w = w // 3
                
                #p_h = h // 6
                #p_w = w // 6

                if h > w:
                    p_w = int((h-w)/2)
                elif h < w:
                    p_h = int((w-h)/2)

                crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                h, w = crop.shape[:2]
                crops.append(crop)
            end_time = time.time()  # [추가] 크롭 종료 시간

            # [추가] 크롭 시간 및 크롭 수 집계
            total_crop_time += (end_time - start_time)
            total_crops += len(crops)
            
            
            os.makedirs(os.path.join(opt.output_path, id), exist_ok=True)
            for j, crop in enumerate(crops):
                cv2.imwrite(os.path.join(opt.output_path, id, "{}_{}.png".format(i, j)), crop)

        # [추가] 크롭 시간 결과 기록
        avg_crop_time = total_crop_time / total_crops if total_crops > 0 else 0
        experiment_content = f'''Preprocessing step: extract_crops
            Total crops: {total_crops}
            Total time: {total_crop_time:.2f}s
            Avg crop time: {avg_crop_time:.6f} s/crop
            '''
            
        with open(os.path.join(RESULT_PATH, 'experiment_content_crops.txt'), 'w', encoding='utf-8') as file:
            file.write(experiment_content)

        if total_crops > 0:
            print(f"\n[추론 시간] 평균 크롭 시간 (CPU): {avg_crop_time:.6f} 초/크롭")
        else:
            print("\n[추론 시간] 처리된 크롭이 없습니다.")

        # if counter == 0:
        #     print(video, counter)
    except e:
        print("Error:", e)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DFDC", type=str,
                        help='Dataset (DFDC / FACEFORENSICS')
    parser.add_argument('--data_path', default='', type=str,
                        help='Videos directory')
    parser.add_argument('--output_path', default='', type=str,
                        help='Output directory')

    opt = parser.parse_args()
    print(opt)
    

    if opt.dataset.upper() == "DFDC":
        dataset = 0
    else:
        dataset = 1
    
    
    os.makedirs(opt.output_path, exist_ok=True)
    #excluded_videos = os.listdir(os.path.join(opt.output_dir)) # Useful to avoid to extract from already extracted videos
    excluded_videos = os.listdir(opt.output_path)
    if dataset == 0:
        paths = get_video_paths(opt.data_path, dataset, excluded_videos)
        #paths.extend(get_video_paths(opt.data_path, dataset, excluded_videos))
    else:
        paths = get_video_paths(os.path.join(opt.data_path, "manipulated_sequences"), dataset)
        paths.extend(get_video_paths(os.path.join(opt.data_path, "original_sequences"), dataset))
    
    with Pool(processes=max(cpu_count()-2, 1)) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(extract_video, root_dir=opt.data_path, dataset=dataset), paths):
                pbar.update()
