import torch
import numpy as np
from datasets import *
from utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

dataset_path = "/cluster/dataset/medinfmk/ARGON/VinDr-CXR"

train_dataset = VinDr_CXR_Dataset(datasetpath=dataset_path, split="train", flag_img = True)
test_dataset = VinDr_CXR_Dataset(datasetpath=dataset_path, split="test", flag_img = True)

dataset_info = [
    {
        "dataset":train_dataset,
        "id_prefix":"vindr-cxr-train",
    }
]
train_llava_dataset = generate_llava_instruction_dataset(dataset_info)


with open("vindr-cxr_detection_train.json", "w") as f:
    json.dump(train_llava_dataset, f, indent=4)