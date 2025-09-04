import json
import os
import sys
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
from torch.utils.data import DataLoader
from utils import process_inference_for_single_instruction, load_llava_with_lora

torch.cuda.empty_cache()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from dataset.utils import *  
from dataset.create_dataset import Chest_ImaGenome_Dataset_regions

model_id = "llava-hf/llava-1.5-7b-hf"  # Hugging Face model ID
processor = AutoProcessor.from_pretrained(model_id)

# Modify checkpoints each time - use relative paths from project root
# checkpoint = './checkpoints/llava-v1.5-7b-task_chest_full_ep1'
checkpoint = './checkpoints/llava-v1.5-7b'
# checkpoint = './checkpoints/llava-v1.5-7b-task_chest_full_all_ep1_multipleInstructions_weighted'
# checkpoint = './checkpoints/llava-v1.5-7b-task_chest_full_all_ep2_multipleInstructions'
# checkpoint = './checkpoints/llava-v1.5-7b-task_chest_full_all_ep3_multipleInstruction'
# checkpoint = './checkpoints/llava-v1.5-7b-task_chest_full_all_ep1_single_instruction'
# checkpoint = './checkpoints/llava-v1.5-7b-task_chest_full_all_ep2_singleInstruction'
# checkpoint = './checkpoints/llava-v1.5-7b-task_chest_full_all_ep3_single_instruction'
# checkpoint = './checkpoints/llava-v1.5-7b-task_chest_full_all_ep1_singleInstruction_weighted'
model, processor2 = load_llava_with_lora(model_id_or_path=model_id, checkpoint_lora_path=checkpoint)

model.half()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Current GPU memory usage after model loading: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

model.eval()

# Update these paths to your local data directories
img_path = "./data/MIMIC-CXR-JPG"  # Path to MIMIC-CXR images
dataset_path = "./data/processed"  # Path to processed dataset files
valid_dataset = Chest_ImaGenome_Dataset_regions(imgpath=img_path, datasetpath=dataset_path, split="validation", flag_img=True)
data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

batch_size_for_saving = 1000 
total_datapoints = len(valid_dataset)

batch_idx = 0
batch_output = []

# modify saved directory each time
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../batches_all_llava_7b')
os.makedirs(output_dir, exist_ok=True)

for idx, data_batch in enumerate(data_loader):
    output = process_inference_for_single_instruction(
        model,
        processor,
        [data_batch], 
        device
    )
    
    for item in output:
        for key, value in item.items():
            if isinstance(value, np.ndarray):
                item[key] = value.tolist()
        if 'img_path' in item:
            item['img'] = item['img_path']

    batch_output.append(output)

    if (idx + 1) % batch_size_for_saving == 0 or (idx + 1) == total_datapoints:
        json_output_path = os.path.join(output_dir, f"eval_output_batch_{batch_idx}_all.json")
        with open(json_output_path, "w") as f:
            json.dump(batch_output, f)
        print(f"Batch {batch_idx} saved successfully to {json_output_path}.")
        batch_output = [] 
        batch_idx += 1
        torch.cuda.empty_cache()
