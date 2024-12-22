import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
from vit_pytorch import ViT
import numpy as np
import os
import json
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from cross_efficient_vit import CrossEfficientViT
from quantized_cross_efficient_vit import QuantizedCrossEfficientViT
import uuid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, RocCurveDisplay
import cv2
from transforms.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim import lr_scheduler
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
# for graphs
import matplotlib.pyplot as plt  
import numpy as np
from sklearn import metrics

# BASE_DIR = '../../deep_fakes/'  
BASE_DIR = '../../deep_fakes/'   ## testing w/ sample data called 'deep_fakes_backup'
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
MODELS_PATH = "models"
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata") # Folder containing all training metadata for DFDC dataset
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")
#
RESULT_PATH = "plots"
BEST_CEV_PATH = os.path.join(MODELS_PATH, "cev_best.pth")
BEST_QCEV_PATH = os.path.join(MODELS_PATH, "qcev_best.pth")

def read_frames(video_path, train_dataset, validation_dataset):
    
    # Get the video label based on dataset selected
    #method = get_method(video_path, DATA_DIR)
    if TRAINING_DIR in video_path:
        if "Original" in video_path:
            label = 0.
        elif "DFDC" in video_path:
            for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                video_folder_name = os.path.basename(video_path)
                video_key = video_folder_name + ".mp4"
                if video_key in metadata.keys():
                    item = metadata[video_key]
                    label = item.get("label", None)
                    if label == "FAKE":
                        label = 1.         
                    else:
                        label = 0.
                    break
                else:
                    label = None
        else:
            label = 1.
        if label == None:
            print("NOT FOUND", video_path)
    else:
        if "Original" in video_path:
            label = 0.
        elif "DFDC" in video_path:
            val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
            video_folder_name = os.path.basename(video_path)
            video_key = video_folder_name + ".mp4"
            label = val_df.loc[val_df['filename'] == video_key]['label'].values[0]
        else:
            label = 1.

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    if label == 0:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-real']),1) # Compensate unbalancing
    else:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-fake']),1)


    
    if VALIDATION_DIR in video_path:
        min_video_frames = int(max(min_video_frames/8, 2))
    frames_interval = int(frames_number / min_video_frames)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0,1):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)
    # Select only the frames at a certain interval
    if frames_interval > 0:
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            
            frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
    # Select N frames from the collected ones
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            #image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
            image = cv2.imread(os.path.join(video_path, frame_image))
            if image is not None:
                if TRAINING_DIR in video_path:
                    train_dataset.append((image, label))
                else:
                    validation_dataset.append((image, label))
                    
# Plotting graphs
'''
def plot_metrics(train_loss_list, val_loss_list): # train_f1_score_list, val_f1_score_list
    epochs = range(1, len(train_loss_list) + 1)  # x축 (epoch)

    # 1. Loss 그래프
    plt.figure(figsize=(12, 6))

    # plt.subplot(2, 2, 1)  # 2행 2열의 첫 번째 그래프
    plt.plot(epochs, train_loss_list, label='Train Loss', color='blue')
    plt.plot(epochs, val_loss_list, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(RESULT_PATH, 'CEV'+ str(t) +'_loss.png'))

    plt.clf()

    # 2. F1 Score 그래프
    # plt.subplot(2, 2, 2)  # 2행 2열의 두 번째 그래프
    plt.plot(epochs, train_f1_score_list, label='Train F1 Score', color='blue')
    plt.plot(epochs, val_f1_score_list, label='Validation F1 Score', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Train and Validation F1 Score')
    plt.legend()
    plt.savefig(os.path.join(RESULT_PATH, 'CEV'+ str(t) + '_f1_score_comparision.png'))

    plt.clf()
    
    # 3. Learning Rate 그래프
    # plt.subplot(2, 2, 3)  # 2행 2열의 세 번째 그래프
    plt.plot(epochs, lr_list, label='Learning Rate', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.legend()

    # # 레이아웃 조정
    # plt.tight_layout()

    # 그래프 이미지 파일로 저장
    plt.savefig(os.path.join(RESULT_PATH, 'CEV'+ str(t) + '_learning_rate.png'))
'''

def roc_auc(y, pred):
    # y = np.array([0, 0, 1, 1])
    # pred = np.array([0.1, 0.4, 0.35, 0.8])
    # calculation
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    auc_score = auc(fpr, tpr)
    # plot the curve
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC Curve')
    display.plot() 
    # plt.show() 
    # save the curve
    # plt.savefig(os.path.join(RESULT_PATH, "CEV_roc_auc_curve.png"))
    plt.savefig(os.path.join(RESULT_PATH, "QCEV_roc_auc_curve.png"))
    return auc_score


# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=5, type=int,  # default = 300 -> 100 -> 5
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=4, type=int,       # default = 10 -> 4
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='Deepfakes', # dafault = All -> Deepfakes
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=10, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    
    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    
    # for regular cross eff vit
    # model = CrossEfficientViT(config=config)
    # model.train()   
    
    # for Quantization
    model = QuantizedCrossEfficientViT(config=config)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1
    else:
        print("No checkpoint loaded.")


    print("Model Parameters:", get_n_params(model))
   
    #READ DATASET
    # if opt.dataset != "All":
    #     folders = ["Original", opt.dataset]
    # else:
    #     folders = ["Original", "DFDC", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    folders = ["DFDC"]   
    
    sets = [TRAINING_DIR, VALIDATION_DIR]

    paths = []
    for dataset in sets:
        for folder in folders:
            subfolder = os.path.join(dataset, folder)
            subfolder_path = os.listdir(subfolder)  ##
            for index, video_folder_name in enumerate(subfolder_path[:len(subfolder_path)//10]): ## 데이터셋 1/10 만 돌리기
            #for index, video_folder_name in enumerate(os.listdir(subfolder)):      # default
                if index == opt.max_videos:
                    break
                if os.path.isdir(os.path.join(subfolder, video_folder_name)):
                    paths.append(os.path.join(subfolder, video_folder_name))
                

    mgr = Manager()
    train_dataset = mgr.list()
    validation_dataset = mgr.list()

    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset),paths):
                pbar.update()
    train_samples = len(train_dataset)
    train_dataset = shuffle_dataset(train_dataset)
    validation_samples = len(validation_dataset)
    validation_dataset = shuffle_dataset(validation_dataset)

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(image[1] for image in train_dataset)
    print(train_counters)
    
    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    print(val_counters)
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    # Create the data loaders
    validation_labels = np.asarray([row[1] for row in validation_dataset])
    labels = np.asarray([row[1] for row in train_dataset])

    train_dataset = DeepFakesDataset(np.asarray([row[0] for row in train_dataset]), labels, config['model']['image-size'])
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    validation_dataset = DeepFakesDataset(np.asarray([row[0] for row in validation_dataset]), validation_labels, config['model']['image-size'], mode='validation')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset
    

    model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*config['training']['bs'])+len(val_dl))
        train_correct = 0
        positive = 0
        negative = 0
        for index, (images, labels) in enumerate(dl):
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.cuda()
            
            y_pred = model(images)
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
            for i in range(config['training']['bs']):
                bar.next()

             
            if index%1200 == 0:
                # print("\nLoss: ", total_loss/counter, "Accuracy: ",train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive)  
                print(f"\n[TRAIN]")
                print(f"(#{t}/{opt.num_epochs}) train_loss: {total_loss/counter:.4f} | "
                f"accuracy: {train_correct/(counter*config['training']['bs']):.4f} | "
                f"\n\ttrain_0s(= REAL): {negative} | "
                f"train_1s(= FAKE): {positive}")
                print("_______________________________________________")

        # Validation
        val_counter = 0
        val_correct = 0
        val_positive = 0
        val_negative = 0
        # for F1-score, Precision, and Recall
        val_preds = []
        val_labels_list = []
        val_prob_list = []
        # for Best.pth
        best_val_loss = 0
       
        train_correct /= train_samples
        total_loss /= counter
        
        for index, (val_images, val_labels) in enumerate(val_dl):
            val_images = np.transpose(val_images, (0, 3, 1, 2))
            val_images = val_images.cuda()
            
            val_labels = val_labels.unsqueeze(1).float()  
            
            val_pred = model(val_images)          # default
            val_pred = val_pred.cpu().float()     # default
            
            val_loss = loss_fn(val_pred, val_labels)
            total_val_loss += round(val_loss.item(), 2)
            
            corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
            val_correct += corrects
            val_positive += positive_class
            val_negative += negative_class
            val_counter += 1
            '''
            # Collect metrics for F1, Precision, Recall, and ROC-AUC
            with torch.no_grad():
                preds = torch.sigmoid(val_pred).numpy() > 0.5  # Binary predictions
                val_preds.extend(preds.flatten())
                val_labels_list.extend(val_labels.cpu().numpy().flatten())
                val_prob_list.extend(torch.sigmoid(val_pred).numpy().flatten())
            '''
            ###
            with torch.no_grad():
                val_pred = model(val_images)  # Get predictions from the model
                val_pred = val_pred.cpu().float()
                preds = torch.sigmoid(val_pred).numpy() > 0.5  # Binary predictions

            # Collect predictions and true labels for precision, recall, and f1-score calculation
            val_preds.extend(preds.flatten())
            val_labels_list.extend(val_labels.cpu().numpy().flatten())
            # for roc_auc curve
            val_prob_list.extend(torch.sigmoid(val_pred).detach().numpy().flatten())
            ###      
            bar.next()
            
            total_val_loss /= val_counter
            val_correct /= validation_samples
            
        scheduler.step()
        bar.finish()
        
        # # # Compute Validation Metrics
        precision = precision_score(val_labels_list, val_preds, average='binary', zero_division=0)
        recall = recall_score(val_labels_list, val_preds, average='binary', zero_division=0)
        f1 = f1_score(val_labels_list, val_preds, average='binary', zero_division=0)
        auc_score = roc_auc(val_labels_list, val_prob_list)

        # Print 
        print(f"\n[VALIDATION]")
        print(f"[Validation Metrics] Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        # Print validation metrics
        print(f"\n[VALIDATION]")
        print(f"[Validation Metrics] Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        val_correct = (np.array(val_preds) == np.array(val_labels_list)).sum()
        val_accuracy = val_correct / len(val_labels_list)
        print(f"[Validation Accuracy]: {val_accuracy:.4f}")
        
        # # #
        # save the best model based on validation loss
        if total_val_loss < previous_loss:
            previous_loss = total_val_loss
            # torch.save(model.state_dict(), BEST_CEV_PATH)   # for regular CEV
            torch.save(model.state_dict(), BEST_QCEV_PATH) # for quantized CEV
            print("*****************************")
            print(f"Best Model saved at epoch {t} ") 
            print("*****************************")
        else:
            not_improved_loss += 1
        
        # print summary
        print(f"(#{t}/{opt.num_epochs}) loss: {total_loss:.4f} | "
                f"accuracy: {train_correct:.4f} | "
                f"\n\tval_loss: {total_val_loss:.4f} | "
                f"val_accuracy: {val_correct:.4f} | "
                f"\n\tval_0s(= REAL): {val_negative}/{np.count_nonzero(validation_labels == 0)} | "
                f"val_1s(= FAKE): {val_positive}/{np.count_nonzero(validation_labels == 1)}"
                )
        print("_______________________________________________")