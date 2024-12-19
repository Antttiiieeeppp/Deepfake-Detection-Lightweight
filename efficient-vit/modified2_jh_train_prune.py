import os
import gc
import json
import glob
import yaml
import time
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

from sklearn.metrics import f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import cv2

from modified2_jh_efficientprune_vit import EfficientViT
from deepfakes_dataset import DeepFakesDataset
from utils import check_correct, shuffle_dataset, get_n_params
 
# 전역 상수 설정
BASE_DIR = "/home/work/Antttiiieeeppp/deep_fakes"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
TEST_DIR = os.path.join(DATA_DIR, "test_set")
MODELS_PATH = "/home/work/Antttiiieeeppp/jaehee/models"
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata")
VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)


class TrainingConfig:
    """학습 설정을 관리하는 클래스"""

    def __init__(self, args, config):
        self.num_epochs = args.num_epochs
        self.batch_size = config["training"]["bs"]
        self.learning_rate = config["training"]["lr"]
        self.weight_decay = config["training"]["weight-decay"]
        self.image_size = config["model"]["image-size"]
        self.workers = args.workers
        self.patience = args.patience
        self.prune_ratio = args.prune_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 현재 시간으로 결과 디렉토리 생성
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.result_path = f"/home/work/Antttiiieeeppp/jaehee/result/{current_time}"
        os.makedirs(self.result_path, exist_ok=True)


class MetricsTracker:
    """학습 중 메트릭 추적"""

    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.train_f1 = []
        self.val_f1 = []
        self.learning_rates = []
        self.best_f1 = -float("inf")
        self.best_epoch = 0

    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_f1: float,
        val_f1: float,
        lr: float,
    ):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_f1.append(train_f1)
        self.val_f1.append(val_f1)
        self.learning_rates.append(lr)

        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.best_epoch = len(self.val_f1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_f1_score: float,
    best_f1_score: float,
    path: str,
):
    """모델 체크포인트 저장"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1_score": val_f1_score,
            "best_f1_score": best_f1_score,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, path: str
) -> Tuple[nn.Module, torch.optim.Optimizer, int, float]:
    """모델 체크포인트 로드"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"], checkpoint["best_f1_score"]


def get_video_label(video_path: str) -> Optional[float]:
    """비디오 레이블 획득"""
    try:
        if TRAINING_DIR in video_path:
            video_folder_name = os.path.basename(video_path)
            video_key = f"{video_folder_name}.mp4"

            for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                if video_key in metadata:
                    return 1.0 if metadata[video_key]["label"] == "FAKE" else 0.0
        else:
            if "Original" in video_path:
                return 0.0
            elif "DFDC" in video_path:
                val_df = pd.read_csv(VALIDATION_LABELS_PATH)
                video_key = f"{os.path.basename(video_path)}.mp4"
                return float(
                    val_df.loc[val_df["filename"] == video_key]["label"].values[0]
                )
            else:
                return 1.0
    except Exception as e:
        logging.error(f"Error getting label for {video_path}: {str(e)}")
    return None


def process_video_frames(
    video_path: str, config: dict
) -> List[Tuple[np.ndarray, float]]:
    """단일 비디오의 프레임들을 처리"""
    frames = []
    label = get_video_label(video_path)
    if label is None:
        return frames

    for frame_path in glob.glob(os.path.join(video_path, "*")):
        try:
            image = cv2.imread(frame_path)
            if image is not None:
                image = cv2.resize(image, (config["model"]["image-size"],) * 2)
                frames.append((image, label))
        except Exception as e:
            logging.warning(f"Error processing frame {frame_path}: {str(e)}")
            continue

    return frames


def batch_process_frames(
    paths: List[str], config: dict, batch_size: int = 100
) -> Tuple[List, List]:
    """
    프레임을 배치 단위로 처리하여 메모리 사용량 최적화
    """
    train_dataset = []
    validation_dataset = []
    current_batch_size = 0

    for path in tqdm(paths, desc="Processing frames"):
        try:
            frames = process_video_frames(path, config)
            if TRAINING_DIR in path:
                train_dataset.extend(frames)
            else:
                validation_dataset.extend(frames)

            current_batch_size += len(frames)

            if current_batch_size >= batch_size:
                yield train_dataset, validation_dataset
                train_dataset = []
                validation_dataset = []
                current_batch_size = 0

        except Exception as e:
            logging.error(f"Error processing video {path}: {str(e)}")
            continue

    if train_dataset or validation_dataset:
        yield train_dataset, validation_dataset


def plot_metrics(metrics_tracker: MetricsTracker, result_path: str):
    """학습 메트릭 시각화"""
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(metrics_tracker.train_loss) + 1)

    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics_tracker.train_loss, label="Train Loss")
    plt.plot(epochs, metrics_tracker.val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # F1 Score plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics_tracker.train_f1, label="Train F1")
    plt.plot(epochs, metrics_tracker.val_f1, label="Validation F1")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("Training and Validation F1 Score")
    plt.legend()

    # Learning rate plot
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics_tracker.learning_rates, label="Learning Rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Over Time")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "training_metrics.png"))
    plt.close()


def plot_roc_curve(
    true_labels: np.ndarray, predictions: np.ndarray, result_path: str
) -> float:
    """ROC 곡선 그리기 및 AUC 계산"""
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_path, "roc_curve.png"))
    plt.close()

    return roc_auc


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: TrainingConfig,
) -> Tuple[float, float, List, List]:
    """단일 에폭 학습"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(progress_bar):
        try:
            # 데이터 전처리
            images = torch.tensor(np.transpose(images.numpy(), (0, 3, 1, 2)))
            images = images.to(config.device)
            labels = labels.unsqueeze(1).float().to(config.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # 메트릭 계산
            total_loss += loss.item()
            predictions.extend(torch.sigmoid(outputs).detach().cpu().numpy().round())
            true_labels.extend(labels.cpu().numpy())

            # 진행 상황 업데이트
            progress_bar.set_description(f"Training - Loss: {loss.item():.4f}")

        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            continue

    avg_loss = total_loss / len(train_loader)
    f1 = f1_score(true_labels, predictions)

    return avg_loss, f1, predictions, true_labels


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: TrainingConfig,
) -> Tuple[float, float, List, List, List]:
    """검증 수행"""
    model.eval()
    total_loss = 0
    predictions = []
    probabilities = []
    true_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        for images, labels in progress_bar:
            try:
                # 데이터 전처리
                images = torch.tensor(np.transpose(images.numpy(), (0, 3, 1, 2)))
                images = images.to(config.device)
                labels = labels.unsqueeze(1).float().to(config.device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 메트릭 계산
                total_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy()
                probabilities.extend(probs)
                predictions.extend(np.round(probs))
                true_labels.extend(labels.cpu().numpy())

                # 진행 상황 업데이트
                progress_bar.set_description(f"Validation - Loss: {loss.item():.4f}")

            except Exception as e:
                logging.error(f"Error during validation: {str(e)}")
                continue

    avg_loss = total_loss / len(val_loader)
    f1 = f1_score(true_labels, predictions)

    return avg_loss, f1, predictions, true_labels, probabilities


def setup_data_loaders(
    config: TrainingConfig, folders: List[str] = ["DFDC"]
) -> Tuple[DataLoader, DataLoader]:
    """데이터 로더 설정"""
    paths = []
    sets = [TRAINING_DIR, VALIDATION_DIR]

    # 데이터셋 경로 수집
    for dataset in sets:
        for folder in folders:
            subfolder = os.path.join(dataset, folder)
            subfolder_path = os.listdir(subfolder)
            for video_folder_name in subfolder_path[: len(subfolder_path) // 100]:
                if os.path.isdir(os.path.join(subfolder, video_folder_name)):
                    paths.append(os.path.join(subfolder, video_folder_name))

    train_dataset = []
    validation_dataset = []

    # 프레임 처리
    for train_batch, val_batch in batch_process_frames(
        paths, config.__dict__, batch_size=100
    ):
        train_dataset.extend(train_batch)
        validation_dataset.extend(val_batch)

    # 데이터셋 셔플
    train_dataset = shuffle_dataset(train_dataset)
    validation_dataset = shuffle_dataset(validation_dataset)

    # 데이터 로더 생성
    train_loader = DataLoader(
        DeepFakesDataset(
            np.asarray([row[0] for row in train_dataset]),
            np.asarray([row[1] for row in train_dataset]),
            config.image_size,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        DeepFakesDataset(
            np.asarray([row[0] for row in validation_dataset]),
            np.asarray([row[1] for row in validation_dataset]),
            config.image_size,
            mode="validation",
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def save_training_results(
    config: TrainingConfig,
    metrics: MetricsTracker,
    training_time: float,
    model: nn.Module,
):
    """학습 결과 저장"""
    results = {
        "train_loss": metrics.train_loss,
        "val_loss": metrics.val_loss,
        "train_f1": metrics.train_f1,
        "val_f1": metrics.val_f1,
        "learning_rates": metrics.learning_rates,
        "best_f1": metrics.best_f1,
        "best_epoch": metrics.best_epoch,
        "training_time": training_time,
        "model_parameters": get_n_params(model),
    }

    # 결과 저장
    with open(os.path.join(config.result_path, "training_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # 실험 내용 저장
    experiment_content = f"""Model_Name: Efficient(Pruning)-ViT
Model_Parameter: {get_n_params(model)}
Epochs: {config.num_epochs}
Batch_Size: {config.batch_size}
Optimizer: SGD
Scheduler: ReduceLROnPlateau
Start_Learning_Rate: {config.learning_rate}
Final_Learning_Rate: {metrics.learning_rates[-1]}

Training Time: {training_time:.2f} seconds
Best Validation F1 Score: {metrics.best_f1:.4f}
Best Epoch: {metrics.best_epoch}

Note: Pruned model training results
"""
    with open(os.path.join(config.result_path, "experiment_content.txt"), "w") as f:
        f.write(experiment_content)


def main():
    # 아규먼트 파서 설정
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs", default=50, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="Number of data loader workers."
    )
    parser.add_argument(
        "--resume", default="", type=str, help="Path to latest checkpoint."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file."
    )
    parser.add_argument(
        "--efficient_net", type=int, default=0, help="EfficientNet version (0 or 7)."
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience."
    )
    parser.add_argument("--prune_ratio", type=float, default=0.5, help="Pruning ratio.")

    opt = parser.parse_args()

    # 설정 로드
    with open(opt.config, "r") as f:
        config = yaml.safe_load(f)

    training_config = TrainingConfig(opt, config)

    # 모델 초기화
    model = EfficientViT(
        config=config,
        channels=1280 if opt.efficient_net == 0 else 2560,
        selected_efficient_net=opt.efficient_net,
    )

    # Pruning 수행
    model.structured_prune_efficientnet(
        layer_indices=[-3, -2, -1], prune_ratio=opt.prune_ratio
    )
    model.to(training_config.device)

    # Optimizer & Scheduler 설정
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, min_lr=1e-6, verbose=True
    )

    criterion = nn.BCEWithLogitsLoss()
    metrics_tracker = MetricsTracker()

    # 체크포인트 로드
    start_epoch = 1
    if os.path.exists(opt.resume):
        model, optimizer, start_epoch, best_f1 = load_checkpoint(
            model, optimizer, opt.resume
        )
        metrics_tracker.best_f1 = best_f1

    # 데이터 로더 설정
    train_loader, val_loader = setup_data_loaders(training_config)

    # 학습 루프
    not_improved = 0
    start_time = time.time()

    try:
        for epoch in range(start_epoch, training_config.num_epochs + 1):
            logging.info(f"Starting epoch {epoch}/{training_config.num_epochs}")

            # 학습
            train_loss, train_f1, train_preds, train_labels = train_epoch(
                model, train_loader, optimizer, criterion, training_config
            )

            # 검증
            val_loss, val_f1, val_preds, val_labels, val_probs = validate(
                model, val_loader, criterion, training_config
            )

            # 메트릭 업데이트
            current_lr = optimizer.param_groups[0]["lr"]
            metrics_tracker.update(train_loss, val_loss, train_f1, val_f1, current_lr)

            # 학습률 조정
            scheduler.step(val_loss)

            # Early stopping 체크
            if val_loss >= metrics_tracker.val_loss[-1]:
                not_improved += 1
                if not_improved >= training_config.patience:
                    logging.info(f"Early stopping triggered after {epoch} epochs")
                    break
            else:
                not_improved = 0

            # 최고 성능 모델 저장
            if val_f1 >= metrics_tracker.best_f1:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_f1,
                    metrics_tracker.best_f1,
                    os.path.join(
                        training_config.result_path,
                        f"best_model_epoch_{epoch}_f1_{val_f1:.4f}.pth",
                    ),
                )

            logging.info(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, LR: {current_lr:.6f}"
            )

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
    finally:
        training_time = time.time() - start_time

        # 결과 저장
        save_training_results(training_config, metrics_tracker, training_time, model)
        plot_metrics(metrics_tracker, training_config.result_path)
        plot_roc_curve(val_labels, val_probs, training_config.result_path)

        logging.info("Training completed")


if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        gc.collect()
        main()
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
