# LLaVA Adaptation for CXRs: Anatomical Region Grounding and Abnormality Detection

*Fine-tuned Large Language and Vision Assistant for anatomical region grounding and abnormality detection in chest X-rays.*

## Overview

This repository contains a fine-tuned version of LLaVA-1.5-7B specifically adapted for medical imaging tasks, focusing on:
- **Anatomical Region Grounding**: Localizing and identifying anatomical structures in chest X-rays
- **Abnormality Detection**: Detecting and localizing pathological findings in medical images

## Training Pipeline

The fine-tuning process follows a structured 5-step pipeline:

### 1. Data Preprocessing 
- Extract objects and anatomical regions from medical imaging datasets
- Generate instruction templates and response patterns
- Process and merge datasets for training consistency

### 2. Dataset Building 
- **Anatomical Region Grounding**: Create instruction-following datasets for anatomical structure localization using Chest ImaGenome dataset
- **Abnormality Detection**: Build VinDr-CXR dataset with instruction templates for pathology detection
- Generate LLaVA-compatible JSON format with conversation pairs

### 3. Fine-tuning 
- Fine-tune LLaVA-1.5-7B using LoRA (Low-Rank Adaptation) for efficient training


### 4. Evaluation 
- Inference with HuggingFace package and performance metrics including mAP, IoU and other metrics for tasks

## Key Features

- **LoRA Fine-tuning**: Memory-efficient training with LoRA adapters
- **Instruction Diversity**: Multiple instruction templates for robust performance

<!-- <a href="https://llava.hliu.cc/"><img src="assets/demo.gif" width="70%"></a> -->

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
**Usage and License Notices**: This project utilizes certain checkpoints that are subject to their respective original licenses. 


## Quick Start

### Virtual Environment
```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Fine-tuning
```bash
# Anatomical region grounding
cd 03_fine_tuning/anatomical_region_grounding/
bash finetune_chestIma_ep1_lora.sh

# Abnormality detection
cd 03_fine_tuning/abnormality_detection/
bash finetune_vindr_ep1_lora.sh
```

### Inference with Hugging Face
The model uses Hugging Face's LLaVA implementation for inference:
```python
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf"  # Base model from Hugging Face
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id)
```

## Dataset Requirements

- **Chest ImaGenome**: For anatomical region grounding
- **VinDr-CXR**: For abnormality detection
- **MIMIC-CXR**: Base chest X-ray images for anatomical region grounding

## Training Configuration

- **Base Model**: LLaVA-1.5-7B git cloned from Github
- **Training Method**: LoRA with rank 128, alpha 256
- **Hardware**: 4x GPU setup with DeepSpeed ZeRO-3
- **Inference**: Hugging Face Transformers library: llava-hf/llava-1.5-7b-hf(https://huggingface.co/llava-hf/llava-1.5-7b-hf)


## Acknowledgments

- Built upon the original [LLaVA](https://github.com/haotian-liu/LLaVA) framework by Liu et al.
- Uses [Hugging Face's LLaVA implementation](https://huggingface.co/llava-hf/llava-1.5-7b-hf) for inference
- Medical datasets: Chest ImaGenome, VinDr-CXR, and MIMIC-CXR

## References


```bibtex
@misc{liu2023visual,
      title={Visual Instruction Tuning}, 
      author={Haotian Liu and Chunyuan Li and Qingyang Wu and Yong Jae Lee},
      year={2023},
      eprint={2304.08485},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This project follows the same license terms as the original LLaVA project. Please refer to the [LICENSE](LICENSE) file for details.
