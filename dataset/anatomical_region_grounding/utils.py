import pandas as pd
import json
import random
import numpy as np


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = [item[key] for item in batch]

    return collated_batch

def capitalize_first_letter(s):
    """ Helper function to capitalize the first letter of a string. """
    stripped = s.lstrip()
    if stripped:
        return stripped[0].upper() + stripped[1:]
    else:
        return s

# Example for testing
# bbox_dict = {"a":[0.23,0.24,0.25,0.26], "b":[1.138,1.288,1.399,1.455,1.566], "c":[2.55,2.66,2.67,2.68]}
np.random.seed(42)

# Function to generate instructions and responses
def generate_instructions_and_responses(bbox_dict):
    # Load instruction and response templates from relative paths
    with open('../../01_preprocess_data/instructions.json', 'r') as f:
        instructions_json = json.load(f)
    with open('../../01_preprocess_data/responses.json', 'r') as f:
        responses_json = json.load(f)
    # Alternative single instruction templates:
    # with open('../../01_preprocess_data/instructions_single.json', 'r') as f:
    #     instructions_json = json.load(f)
    # with open('../../01_preprocess_data/responses_single.json', 'r') as f:
    #     responses_json = json.load(f)

    anatomies = list(bbox_dict.keys())
    coordinates = [f"[{round(coord[0], 2)}, {round(coord[1], 2)}, {round(coord[2], 2)}, {round(coord[3], 2)}]" for coord in bbox_dict.values()]

    num_anatomies = len(anatomies)
    if num_anatomies > 5:
        num_anatomies = 5 

    instruction_key = f"{num_anatomies}_anatomies"
    instructions = instructions_json["visual grounding"].get(instruction_key, [])
    responses = responses_json.get(instruction_key, [])

    if instructions and responses:
        selected_instruction = random.choice(instructions)
        selected_response = random.choice(responses)

        anatomy_placeholders = {f'anatomy{i+1}': anatomies[i] for i in range(num_anatomies)}
        coordinates_placeholders = {f'coordinates{i+1}': coordinates[i] for i in range(num_anatomies)}

        formatted_instruction = selected_instruction.format(**anatomy_placeholders)
        formatted_response = selected_response.format(**anatomy_placeholders, **coordinates_placeholders)
        formatted_instruction = capitalize_first_letter(formatted_instruction)
        formatted_response = capitalize_first_letter(formatted_response)

        instruction = {
            'question': formatted_instruction,
            'answer': formatted_response
        }
        return instruction

def generate_llava_instruction_dataset(dataset_info, seed=0):
    json_structure = []
    for dataset_i, dataset_info_cell in enumerate(dataset_info):
        np.random.seed(seed)
        random.seed(seed)

        if "id_prefix" not in dataset_info_cell:
            dataset_info_cell["id_prefix"] = dataset_i
        
        idx = range(len(dataset_info_cell["dataset"]))
        if "num_samples" in dataset_info_cell:
            idx = random.sample(idx, dataset_info_cell["num_samples"])
        
        for i, id_cell in enumerate(idx):
            sample = dataset_info_cell["dataset"][id_cell]
            sample_instr = sample.get("instr") 

            if sample_instr is None:
                print(f"Warning: No instructions found for sample at index {id_cell}. Skipping.")
                continue

            json_cell = {
                "image": sample["img_path"],
                "conversations": [],
                "id": f"{dataset_info_cell['id_prefix']}_{i}"
            }

            if isinstance(sample_instr, dict):
                sample_instr = [sample_instr]

            for j, instr in enumerate(sample_instr):
                conv_cell_human = {
                    "from": "human",
                    "value": instr["question"]
                }
                if j == 0:
                    conv_cell_human["value"] = f"<image>\n{conv_cell_human['value']}"

                conv_cell_ai = {
                    "from": "gpt",
                    "value": instr["answer"]
                }
                json_cell["conversations"].append(conv_cell_human)
                json_cell["conversations"].append(conv_cell_ai)
            
            json_structure.append(json_cell)

    return json_structure
