#!/bin/bash
 
#SBATCH --job-name=llava_chest
#SBATCH --output=chestima_dataset_multipleInstructions_weighted.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=128G
 
source ~/.bashrc
conda activate eval_llava
# pip install numpy

# python generate_instructions.py
# python demo.py
python create_llava_dataset.py
