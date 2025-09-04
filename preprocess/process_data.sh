#!/bin/bash
 
#SBATCH --job-name=llava_data
#SBATCH --output=processed_dataset.txt
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=128G
 
source ~/.bashrc
conda activate eval_llava
 
# python extract_objects_original.py
# python build_instructions.py --csv llava_train.csv --output train.json --instructions instructions.json --responses responses.json
# python build_instructions.py --csv llava_test.csv --output test.json --instructions instructions.json --responses responses.json
# python build_instructions.py --csv llava_valid.csv --output valid.json --instructions instructions.json --responses responses.json
# python test_path.py
# python count_anatomy.py
# python merge_select.py
# python build_instructions.py --csv data/llava_train.csv --output train_single_instruction.json --instructions instructions_single.json --responses responses_single.json --csv_output llava_train_single.csv
# python build_instructions.py --csv data/llava_test.csv --output test_single_instructio.json --instructions instructions_single.json --responses responses_single.json --csv_output llava_test_single.csv
# python build_instructions.py --csv data/llava_valid.csv --output valid_single_instruction.json --instructions instructions_single.json --responses responses_single.json --csv_output llava_test_single.csv
# python merge_select_weighted.py
python count_anatomy.py
