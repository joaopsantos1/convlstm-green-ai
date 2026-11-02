import os
from datetime import datetime
import numpy as np
from torch import Tensor
from torchvision.utils import save_image

CONVOLUTION_NAME = "standard"


def build_root_folder_name(
    step_name: str, dataset_name: str, folder_suffix: str
) -> str:
    suffix_name = (
        folder_suffix
        if folder_suffix != ""
        else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    root_folder = f"{step_name}_{dataset_name}_{CONVOLUTION_NAME}_{suffix_name}"
    return root_folder


def create_training_working_directory(
    working_dir: str,
    dataset_name: str,
    train_folder_suffix: str,
    model_architecture: str = "convlstm",
) -> dict:
    if model_architecture != "convlstm" and not train_folder_suffix.startswith(
        f"{model_architecture}_"
    ):
        timestamp = train_folder_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        train_folder_suffix = f"{model_architecture}_{timestamp}"
    training_root_folder = build_root_folder_name(
        "training", dataset_name, train_folder_suffix
    )
    main_dir = f"{working_dir}/{training_root_folder}"
    weights_dir = f"{main_dir}/weights"

    training_dirs = {
        "metrics": f"{main_dir}/metrics",
        "weights": {
            "model": f"{weights_dir}/{model_architecture}",
        },
        "emissions": f"{main_dir}/emissions",
    }

    os.makedirs(main_dir, exist_ok=True)
    os.mkdir(training_dirs["metrics"])
    os.mkdir(weights_dir)
    os.mkdir(training_dirs["weights"]["model"])
    os.mkdir(training_dirs["emissions"])

    return training_dirs

def create_predict_working_directory(
    working_dir: str,
    dataset_name: str,
    predict_folder_suffix: str,
    model_architecture: str = "convlstm",
) -> dict:
    if model_architecture != "convlstm" and not predict_folder_suffix.startswith(
        f"{model_architecture}_"
    ):
        timestamp = predict_folder_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        predict_folder_suffix = f"{model_architecture}_{timestamp}"
    predict_root_folder = build_root_folder_name(
        "predict", dataset_name, predict_folder_suffix
    )
    main_dir = f"{working_dir}/{predict_root_folder}"
    images_dir = f"{main_dir}/images"
    npy_files_dir = f"{main_dir}/npy_files"

    predict_dirs = {
        "metrics": f"{main_dir}/metrics",
        "images": {
            "real": f"{images_dir}/real",
            "generated": f"{images_dir}/generated",
        },
        "npy_files": {
            "real": f"{npy_files_dir}/real",
            "generated": f"{npy_files_dir}/generated",
        },
        "emissions": f"{main_dir}/emissions",
    }

    os.makedirs(main_dir, exist_ok=True)
    os.mkdir(predict_dirs["metrics"])
    os.mkdir(images_dir)
    os.mkdir(predict_dirs["images"]["real"])
    os.mkdir(predict_dirs["images"]["generated"])
    os.mkdir(npy_files_dir)
    os.mkdir(predict_dirs["npy_files"]["real"])
    os.mkdir(predict_dirs["npy_files"]["generated"])
    os.mkdir(predict_dirs["emissions"])

    return predict_dirs


def save_sequence_images(
    batch_seq_images: Tensor,
    batch_size: int,
    saving_dir: str,
    step: int,
):
    for seq_idx in range(batch_size):
        save_image(
            batch_seq_images[seq_idx],
            f"{saving_dir}/{CONVOLUTION_NAME}_sequence_{step}_{seq_idx + 1}.png",
        )


def save_sequence_npy(
    batch_seq_images: Tensor, saving_dir: str, step: int
):
    np.save(
        f"{saving_dir}/{CONVOLUTION_NAME}_sequence_{step}.npy",
        batch_seq_images.cpu().numpy(),
    )
