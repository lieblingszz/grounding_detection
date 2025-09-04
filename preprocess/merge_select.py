import pandas as pd
import numpy as np
import os
from pathlib import Path

np.random.seed(42)

files = ['/cluster/home/jingma/data_processing/data/llava_train_01.csv', '/cluster/home/jingma/data_processing/data/llava_valid_01.csv', '/cluster/home/jingma/data_processing/data/llava_test_01.csv']

df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
# print(df.head())



def select_random_bbox(sub_df):
    num_samples = np.random.randint(1, 6)  
    return sub_df.sample(n=min(num_samples, len(sub_df)))  


df = df.groupby('image_id').apply(select_random_bbox).reset_index(drop=True)

prefix = '/cluster/work/medinfmk/ARGON/MIMIC-CXR-JPG/'


df['full_image_path'] = prefix + df['image_path'].astype(str)
df['image_exists'] = df['full_image_path'].apply(lambda path: os.path.exists(path))


missing_images = df[df['image_exists'] == False]['image_id'].unique()
if len(missing_images) > 0:
    print("Image IDs with missing images:")
    for image_id in missing_images:
        print(image_id)

df = df[df['image_exists']].drop(columns=['image_exists', 'full_image_path'])  
df.to_csv('filtered_dataset.csv', index=False)

print("Preview of filtered DataFrame:")
print(df.head())
