import os
import os.path
import sys 


import tarfile, xml
import numpy as np
import pandas as pd
import skimage
import string
from typing import Dict, List
import skimage.transform
from skimage.io import imread
import random
import torch
from torch.utils.data import DataLoader

import json
from torchxrayvision.datasets import Dataset, normalize
from PIL import Image
from imageio import imread
# from dataset.utils import generate_instructions_and_responses

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.utils import generate_instructions_and_responses



class Chest_ImaGenome_Dataset_regions(Dataset):
    def __init__(self, imgpath, datasetpath, split="train", flag_img=True, flag_instr=True, seed=0):
        super(Chest_ImaGenome_Dataset_regions, self).__init__()
        np.random.seed(seed)
        self.imgpath = imgpath
        self.flag_img = flag_img
        self.flag_instr = flag_instr
        # revise
        full_data_path = os.path.join(datasetpath, "filtered_dataset.csv")
        self.data = pd.read_csv(full_data_path)
        self.data = self.data[self.data['split'] == split]
        self.grouped_data = {k: v for k, v in self.data.groupby('image_id')}
        self.indexed_data = list(self.grouped_data.items())

    def __len__(self):
        return len(self.indexed_data)

    def __getitem__(self, idx):
        image_id, group = self.indexed_data[idx]
        sample = {"idx": idx, "image_id": image_id}
        img_path = os.path.join(self.imgpath, group['image_path'].iloc[0])
        sample["img_path"] = img_path

        if self.flag_img:
            img = imread(img_path)
            sample["img"] = normalize(img, maxval=255, reshape=True)

        bbx_dict = {
            row['bbox_name']: [
                round(float(row['x1']), 2),
                round(float(row['y1']), 2),
                round(float(row['x2']), 2),
                round(float(row['y2']), 2)
            ]
            for _, row in group.iterrows()
        }

        sample["bbx_dict"] = bbx_dict

        if self.flag_instr:
            if not bbx_dict:  
                print(f"BBX Dict is empty for sample {idx}")
            else:
                print(f"BBX Dict before processing for sample {idx}: {bbx_dict}")
            instructions = generate_instructions_and_responses(bbx_dict)
            if instructions and isinstance(instructions, dict):
                sample["instr"] = instructions
            else:
                # Provide a default instruction if the returned instruction is not a dictionary
                sample["instr"] = {'question': 'No valid instruction generated.', 'answer': 'No valid answer available.'}
                print(f"Warning: Invalid instruction format for sample {idx}, provided default instruction.")
        return sample