from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
from vit_pytorch import ViT
import numpy as np
from torch.optim import lr_scheduler
import os
import json
from os import cpu_count
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from efficient_vit import EfficientViT
from hw_EfficientNextVit import EfficientNextViT
from cnn import SimpleCNN
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
import cv2
from transforms.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import LambdaLR
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
import gc
from sklearn.metrics import f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
import time

torch.cuda.empty_cache()
gc.collect()

BASE_DIR = "../../deep_fakes/"
# BASE_DIR = '../../deep_fakes_backup/'
# BASE_DIR = '../../deep_fakes_tmp/'
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
MODELS_PATH = "models"
METADATA_PATH = os.path.join(
    BASE_DIR, "data/metadata"
)  # Folder containing all training metadata for DFDC dataset
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")

# 현재 시간으로 디렉토리 이름 생성
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RESULT_PATH = f"result/{current_time}"

# 디렉토리 생성
os.makedirs(RESULT_PATH, exist_ok=True)


def read_frames(video_path, train_dataset, validation_dataset):
    # Get the video label based on dataset selected
    # method = get_method(video_path, DATA_DIR)
    # print(TRAINING_DIR)
    # print(video_path)
    if TRAINING_DIR in video_path:
        # if "Original" in video_path:
        #     label = 0.
        # elif "DFDC" in video_path:
        for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
            with open(json_path, "r") as f:
                metadata = json.load(f)
            video_folder_name = os.path.basename(video_path)
            video_key = video_folder_name + ".mp4"
            if video_key in metadata.keys():

                item = metadata[video_key]

                label = item.get("label", None)
                if label == "FAKE":
                    label = 1.0
                else:
                    label = 0.0
                break
            else:
                label = None
        # else:
        # label = 1.
        if label == None:
            print("NOT FOUND", video_path)
    else:
        if "Original" in video_path:
            label = 0.0
        elif "DFDC" in video_path:
            val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
            video_folder_name = os.path.basename(video_path)
            video_key = video_folder_name + ".mp4"  # <- Original

            # video_key = video_folder_name

            label = val_df.loc[val_df["filename"] == video_key]["label"].values[0]

        else:
            label = 1.0

    for frame_path in glob.glob(os.path.join(video_path, "*")):
        image = cv2.imread(frame_path)
        if image is not None:
            # Resize the image to a fixed size
            image = cv2.resize(
                image, (config["model"]["image-size"], config["model"]["image-size"])
            )
            # Add the resized image to the dataset
            if TRAINING_DIR in video_path:
                train_dataset.append((image, label))
            else:
                validation_dataset.append((image, label))

    # Calculate the interval to extract the frames
    # frames_number = len(os.listdir(video_path)) # 각 .mp4 영상에 존재하는 프레임 수
    # print(f"frames_number: {frames_number}")
    # if label == 0:
    #     min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing_real']), 1) # Compensate unbalancing
    # else:
    #     min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing_fake']), 1)
    # # print(f"min_video_frames: {min_video_frames}")

    # if VALIDATION_DIR in video_path:
    #     min_video_frames = int(max(min_video_frames/8, 2))
    # frames_interval = int(frames_number / min_video_frames)
    # frames_paths = os.listdir(video_path)
    # frames_paths_dict = {}

    # # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    # for path in frames_paths:
    #     for i in range(0,1):
    #         if "_" + str(i) in path:
    #             if i not in frames_paths_dict.keys():
    #                 frames_paths_dict[i] = [path]
    #             else:
    #                 frames_paths_dict[i].append(path)

    # # Select only the frames at a certain interval
    # if frames_interval > 0:
    #     for key in frames_paths_dict.keys():
    #         if len(frames_paths_dict) > frames_interval:
    #             frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]

    #         frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]

    # # # Select N frames from the collected ones
    # # for key in frames_paths_dict.keys():
    # #     for index, frame_image in enumerate(frames_paths_dict[key]):
    # #         #image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
    # #         image = cv2.imread(os.path.join(video_path, frame_image))
    # #         if image is not None:
    # #             if TRAINING_DIR in video_path:
    # #                 train_dataset.append((image, label))
    # #             else:
    # #                 validation_dataset.append((image, label))

    # # Select N frames from the collected ones
    # for key in frames_paths_dict.keys():
    #     for index, frame_image in enumerate(frames_paths_dict[key]):
    #         # Read the frame
    #         image = cv2.imread(os.path.join(video_path, frame_image))
    #         if image is not None:
    #             # Resize the image to a fixed size
    #             image = cv2.resize(image, (config['model']['image-size'], config['model']['image-size']))
    #             # Add the resized image to the dataset
    #             if TRAINING_DIR in video_path:
    #                 train_dataset.append((image, label))
    #             else:
    #                 validation_dataset.append((image, label))


# 그래프 그리기 함수
def plot_metrics(
    train_loss_list, val_loss_list, lr_list, train_f1_score_list, val_f1_score_list
):
    epochs = range(1, len(train_loss_list) + 1)  # x축 (epoch)

    # 1. Loss 그래프
    plt.figure(figsize=(12, 6))

    # plt.subplot(2, 2, 1)  # 2행 2열의 첫 번째 그래프
    plt.plot(epochs, train_loss_list, label="Train Loss", color="blue")
    plt.plot(epochs, val_loss_list, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(RESULT_PATH, "loss.png"))

    plt.clf()

    # 2. F1 Score 그래프
    # plt.subplot(2, 2, 2)  # 2행 2열의 두 번째 그래프
    plt.plot(epochs, train_f1_score_list, label="Train F1 Score", color="blue")
    plt.plot(epochs, val_f1_score_list, label="Validation F1 Score", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Train and Validation F1 Score")
    plt.legend()
    plt.savefig(os.path.join(RESULT_PATH, "f1_score.png"))

    plt.clf()

    # 3. Learning Rate 그래프
    # plt.subplot(2, 2, 3)  # 2행 2열의 세 번째 그래프
    plt.plot(epochs, lr_list, label="Learning Rate", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate")
    plt.legend()

    # # 레이아웃 조정
    # plt.tight_layout()

    # 그래프 이미지 파일로 저장
    plt.savefig(os.path.join(RESULT_PATH, "learning_rate.png"))


def plot_roc_auc(val_labels_list, val_preds):
    # ROC 곡선 계산
    fpr, tpr, thresholds = roc_curve(val_labels_list, val_preds)
    roc_auc = auc(fpr, tpr)  # AUC 값 계산

    # ROC 곡선 그리기
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # 무작위 예측선
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    # 그래프 저장
    plt.savefig(os.path.join(RESULT_PATH, "roc_auc_curve.png"))
    return roc_auc


# Main body
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs", default=50, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="Number of data loader workers."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="Path to latest checkpoint (default: none).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="All",
        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=-1,
        help="Maximum number of videos to use for training (default: all).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Which configuration to use. See into 'config' folder.",
    )
    parser.add_argument(
        "--efficient_net",
        type=int,
        default=0,
        help="Which EfficientNet version to use (0 or 7, default: 0)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="How many epochs wait before stopping for validation loss not improving.",
    )
    opt = parser.parse_args()
    print(f"opt : {opt.config}")
    with open(opt.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    if opt.efficient_net == 0:
        channels = 1280
    else:
        channels = 2560
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # model = EfficientViT(config=config, channels=channels, selected_efficient_net = opt.efficient_net)
    # model = SimpleCNN(device=device)
    # model = EfficientNextViT(num_classes=1, efficientnet_type='efficientnet-b0', head_depths=[2, 2, 6, 2], head_dim=32)
    config2 = {"model": {"num-classes": 1}}
    model = EfficientNextViT(config=config2, selected_efficient_net=0)
    model.train()
    print(f"config: {config}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight-decay"],
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    print(f"optimizer: {optimizer}")
    # scheduler = lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=config["training"]["step-size"],
    #     gamma=config["training"]["gamma"],
    # )
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    starting_epoch = 1
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = (
            int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1
        )  # The checkpoint's file name format should be "checkpoint_EPOCH"
    else:
        print("No checkpoint loaded.")

    print("Model Parameters:", get_n_params(model))

    # READ DATASET
    # if opt.dataset != "All" and opt.dataset != "DFDC":
    #     folders = ["Original", opt.dataset]
    # else:
    #     folders = ["Original", "DFDC", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    folders = ["DFDC"]

    sets = [TRAINING_DIR, VALIDATION_DIR]

    paths = []
    for dataset in sets:
        for folder in folders:
            subfolder = os.path.join(dataset, folder)
            for index, video_folder_name in enumerate(os.listdir(subfolder)):
                if index == opt.max_videos:
                    break

                if os.path.isdir(os.path.join(subfolder, video_folder_name)):
                    paths.append(os.path.join(subfolder, video_folder_name))

    # mgr = Manager()
    # train_dataset = mgr.list()
    # validation_dataset = mgr.list()

    train_dataset = []
    validation_dataset = []

    for path in tqdm(paths):
        read_frames(
            path, train_dataset=train_dataset, validation_dataset=validation_dataset
        )
    # with Pool(processes=opt.workers) as p:
    #     with tqdm(total=len(paths)) as pbar:
    #         for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset),paths):
    #             pbar.update()

    train_samples = len(train_dataset)
    train_dataset = shuffle_dataset(train_dataset)
    validation_samples = len(validation_dataset)
    validation_dataset = shuffle_dataset(validation_dataset)

    # Print some useful statistics
    print(
        "Train images:",
        len(train_dataset),
        "Validation images:",
        len(validation_dataset),
    )
    print("__TRAINING STATS__")
    train_counters = collections.Counter(image[1] for image in train_dataset)
    print(train_counters)

    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    print(val_counters)
    print("___________________")

    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Create the data loaders
    validation_labels = np.asarray([row[1] for row in validation_dataset])
    labels = np.asarray([row[1] for row in train_dataset])

    train_dataset = DeepFakesDataset(
        np.asarray([row[0] for row in train_dataset]),
        labels,
        config["model"]["image-size"],
    )
    dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["bs"],
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=opt.workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=2,
        persistent_workers=False,
    )
    del train_dataset

    validation_dataset = DeepFakesDataset(
        np.asarray([row[0] for row in validation_dataset]),
        validation_labels,
        config["model"]["image-size"],
        mode="validation",
    )
    val_dl = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config["training"]["bs"],
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=opt.workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=2,
        persistent_workers=False,
    )
    del validation_dataset

    # model = model.cuda()
    model = model.to(device)
    print("=-=-=-=-=-=")
    print(model)
    print("=-=-=-=-=-=")
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf

    train_loss_list = []
    val_loss_list = []
    lr_list = []
    train_f1_score_list = []
    val_f1_score_list = []

    start_time = time.time()

    inference_time_list = []
    for t in tqdm(range(starting_epoch, opt.num_epochs + 1)):
        # for t in range(starting_epoch, opt.num_epochs):
        # if not_improved_loss == opt.patience:
        #     break
        counter = 0

        total_loss = 0
        total_val_loss = 0

        bar = ChargingBar(
            "EPOCH #" + str(t), max=(len(dl) * config["training"]["bs"]) + len(val_dl)
        )
        train_correct = 0
        positive = 0
        negative = 0
        train_preds = []  # 훈련 예측값 저장 리스트
        train_labels_list = []  # 훈련 실제 레이블 저장 리스트

        for index, (images, labels) in enumerate(dl):
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            # images = images.cuda()
            images = images.to(device)

            inference_start = time.time()
            y_pred = model(images)
            inference_time_list.append(round(time.time() - inference_start, 4))

            y_pred = y_pred.cpu()

            loss = loss_fn(y_pred, labels)

            corrects, positive_class, negative_class = check_correct(y_pred, labels)
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)

            # Predicted labels for F1 score calculation
            train_preds.extend(
                torch.sigmoid(y_pred).detach().numpy().round()
            )  # 예측값 추가
            train_labels_list.extend(labels.cpu().numpy())  # 실제 레이블 추가

            # if index%1200 == 0: # Intermediate metrics print
            #     print("\nLoss: ", round(total_loss/counter, 4), "Accuracy: ",round(train_correct/(counter*config['training']['bs']), 4) ,"Train 0s: ", negative, "Train 1s:", positive)

            for i in range(config["training"]["bs"]):
                bar.next()

        val_correct = 0
        val_positive = 0
        val_negative = 0
        val_counter = 0

        val_preds = []  # 검증 예측값 저장 리스트
        val_prob_list = []  # 검증 데이터에 대한 예측 확률 리스트
        val_labels_list = []  # 검증 실제 레이블 저장 리스트

        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_labels) in enumerate(val_dl):

            val_images = np.transpose(val_images, (0, 3, 1, 2))

            # val_images = val_images.cuda()
            val_images = val_images.to(device)
            val_labels = val_labels.unsqueeze(1).float()

            val_pred = model(val_images)
            val_pred = val_pred.cpu().float()

            val_loss = loss_fn(val_pred, val_labels)
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(
                val_pred, val_labels
            )
            val_correct += corrects
            val_positive += positive_class
            val_counter += 1
            val_negative += negative_class

            val_prob_list.extend(torch.sigmoid(val_pred).detach().numpy().flatten())

            val_preds.extend(
                torch.sigmoid(val_pred).detach().numpy().round()
            )  # 예측값 추가
            val_labels_list.extend(val_labels.cpu().numpy())  # 실제 레이블 추가

            # print(val_prob_list)
            # print(val_labels_list)

            bar.next()

        bar.finish()

        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0

        scheduler.step(total_val_loss)

        previous_loss = total_val_loss

        print(
            "#"
            + str(t)
            + "/"
            + str(opt.num_epochs)
            + " loss:"
            + str(round(total_loss, 4))
            + " accuracy:"
            + str(round(train_correct, 4))
            + " val_loss:"
            + str(round(total_val_loss, 4))
            + " val_accuracy:"
            + str(round(val_correct, 4))
            + " val_0s:"
            + str(val_negative)
            + "/"
            + str(np.count_nonzero(validation_labels == 0))
            + " val_1s:"
            + str(val_positive)
            + "/"
            + str(np.count_nonzero(validation_labels == 1))
        )

        train_loss_list.append(1 if round(total_loss, 4) >= 1 else round(total_loss, 4))
        val_loss_list.append(round(total_val_loss, 4))
        lr_list.append(scheduler.get_last_lr()[0])

        train_f1_score = f1_score(train_labels_list, train_preds)  # F1 score 계산
        train_f1_score_list.append(round(train_f1_score, 4))  # F1 score 리스트에 추가

        val_f1_score = round(
            f1_score(val_labels_list, val_preds, zero_division=1), 4
        )  # 검증 F1 score 계산
        val_f1_score_list.append(val_f1_score)  # 검증 F1 score 리스트에 추가

        print(f"train_loss_list: {train_loss_list}")
        print(f"val_loss_list: {val_loss_list}")
        print(f"lr_list: {lr_list}")
        print(f"train_f1_score_list: {train_f1_score_list}")
        print(f"val_f1_score_list: {val_f1_score_list}")
        # print(f"inference_time_list: {inference_time_list}")
        if val_f1_score == max(val_f1_score_list):
            torch.save(
                model.state_dict(),
                os.path.join(RESULT_PATH, "efficientnetB" + str(opt.efficient_net)),
            )

    training_time = time.time() - start_time
    # if not os.path.exists(MODELS_PATH):
    #     os.makedirs(MODELS_PATH)
    # torch.save(model.state_dict(), os.path.join(MODELS_PATH,  "efficientnetB"+str(opt.efficient_net)+"_checkpoint" + str(t) + "_" + opt.dataset))
    plot_metrics(
        train_loss_list, val_loss_list, lr_list, train_f1_score_list, val_f1_score_list
    )

    # AUC와 ROC 그래프 그리기
    auc = plot_roc_auc(val_labels_list, val_prob_list)

    # 파일에 쓸 내용
    experiment_content = f"""Model_Name: Efficient-ViT
Model_Parameter:{get_n_params(model)}
Device: {device}
Epoch: {opt.num_epochs}
Batch_Size: {config['training']['bs']}
Optimizer: Adamw
Scheduler: StepLR
Start_Learning_Rate: {config['training']['lr']}
End_Learning_Rate: {scheduler.get_last_lr()[0]}

Training Time: {training_time}
Best Validation f1_score: {max(val_f1_score_list)}
AUC: {auc}
AVG Inference Time: {sum(inference_time_list) / len(inference_time_list)}

Note: 추론 시간을 측정하기 위한 테스트
"""
    #     =-=-=-=-=-=-=-=
    # 파일을 쓰기 모드로 열기 (파일이 없으면 새로 생성됨)
    with open(
        os.path.join(RESULT_PATH, "experiment_content.txt"), "w", encoding="utf-8"
    ) as file:
        file.write(experiment_content)
