import torch
import numpy as np
from create_dataset import *
from utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json

# Update these paths to your local data directories
dataset_path = "./data/processed"  # Path to processed dataset files
img_path = "./data/MIMIC-CXR-JPG"  # Path to MIMIC-CXR images
train_dataset = Chest_ImaGenome_Dataset_regions(imgpath=img_path, datasetpath=dataset_path, split="train", flag_img = False, flag_instr=True)
valid_dataset = Chest_ImaGenome_Dataset_regions(imgpath=img_path, datasetpath=dataset_path, split="validation", flag_img = False, flag_instr=True)
test_dataset = Chest_ImaGenome_Dataset_regions(imgpath=img_path, datasetpath=dataset_path, split="test", flag_img = False, flag_instr=True)

# data_loader = DataLoader(dataset, batch_size = 10, shuffle = True, collate_fn=custom_collate_fn)
# train_len = len(train_dataset)
# valid_len = len(valid_dataset)
# test_len = len(test_dataset)

# print(f"Training dataset size: {train_len}")
# print(f"Validation dataset size: {valid_len}")
# print(f"Test dataset size: {test_len}")

# Start from the small dataset
dataset_info = [
    {
        "dataset":train_dataset,
        "id_prefix":"chestima-train"
    }
]

train_llava_dataset = generate_llava_instruction_dataset(dataset_info)
with open("chest_ima_train_full_multipleInstructions_weighted.json", "w") as f:
    json.dump(train_llava_dataset, f, indent=4)
print('Filese saved')


