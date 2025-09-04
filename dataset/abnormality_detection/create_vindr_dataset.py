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
from torchxrayvision.datasets import Dataset, normalize, apply_transforms,USE_INCLUDED_FILE, Openi_Dataset, RSNA_Pneumonia_Dataset, NIH_Dataset, NIH_Google_Dataset, CheX_Dataset
from PIL import Image
from imageio import imread

# Add the parent directory to the sys.path to allow absolute imports to work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from health_mm_llm_data.create_instructions import *
from health_mm_llm_data.utils import * 


class VinDr_CXR_Dataset(Dataset):
    def __init__(self, datasetpath, split="train", flag_img=True, flag_instr=True, seed=0):
        super(VinDr_CXR_Dataset, self).__init__()
        np.random.seed(seed)
        self.datasetpath = datasetpath
        self.imgpath = os.path.join(self.datasetpath, split + '_jpg')
        self.flag_img = flag_img
        self.flag_instr = flag_instr

        self.image_files = [os.path.splitext(file)[0] for file in os.listdir(self.imgpath) if file.endswith('.jpg')]

        annotations_dir = 'annotations' if os.path.isdir(self.datasetpath + '/annotations') else ''
        annotations_path = os.path.join(self.datasetpath, annotations_dir, f'annotations_{split}.csv')
        self.annotations = pd.read_csv(annotations_path)

        resolutions_path = os.path.join(self.datasetpath, annotations_dir, 'image_resolutions_' + split + '.json')
        if not os.path.exists(resolutions_path):
            raise ValueError("The image resolutions file cannot be found.")
        with open(resolutions_path, 'r') as file:
            self.resolutions = json.load(file)

        if split == 'train':
            self.annotations.rename(columns={'rad_ID': 'rad_id'}, inplace=True)
            
            rad_ids = self.annotations.groupby('image_id')['rad_id'].unique().apply(lambda x: np.random.choice(x))
            # Filter the annotations to include only those from the selected radiologist
            self.annotations['selected_rad'] = self.annotations['image_id'].map(rad_ids)
            self.annotations = self.annotations[self.annotations['rad_id'] == self.annotations['selected_rad']]
            self.image_to_annotations = self.annotations.set_index('image_id')
        elif split == 'test':
            self.image_to_annotations = self.annotations.set_index('image_id')

            

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        sample = {}
        image_id = str(self.image_files[idx])
        img_filename = image_id + '.jpg'
        imgpath = os.path.join(self.imgpath, img_filename)
        sample["img_path"] = imgpath

        if self.flag_img:
            img =  imread(imgpath)
            sample["img"] = normalize(img, maxval=255, reshape=True)

        original_resolution = self.resolutions.get(image_id, (1, 1))

        if image_id in self.image_to_annotations.index:
            image_annotations = self.image_to_annotations.loc[image_id]
            if not isinstance(image_annotations, pd.DataFrame):
                image_annotations = image_annotations.to_frame().T  # Handle single row

            bounding_boxes = []
            class_labels = []
            for _, row in image_annotations.iterrows():
                if row['class_name'] != "No finding":
                    x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']]
                    bounding_boxes.append([
                        round(x_min / original_resolution[0], 3),
                        round(y_min / original_resolution[1], 3),
                        round(x_max / original_resolution[0], 3),
                        round(y_max / original_resolution[1], 3),
                    ])
                    class_labels.append(row["class_name"])

            if not bounding_boxes:  # No abnormalities detected
                bounding_boxes = []
                class_labels = ["No finding"]
        else:
            bounding_boxes = []
            class_labels = ["No finding"]

        sample["boxes"] = bounding_boxes
        sample["labels"] = class_labels

        if self.flag_instr:
            sample["instr"] = generate_instruction_abnormalities_grouped(bounding_boxes, class_labels)

        return sample