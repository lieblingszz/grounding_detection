import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from peft import PeftModel
from numpy import asarray
import safetensors.torch
import os
import json
import pandas as pd

def process_inference_for_single_instruction(model, processor, data_loader, device, user_prompt="USER: <image>\n", assistant_prompt="ASSISTANT:", max_new_token=200, process_batch_num=None, **kwargs):
    """
    Process inference for a single instruction.

    Args:
        model (object): The model used for inference.
        processor (object): The processor used for data preprocessing.
        data_loader (object): The data loader containing the input data.
        user_prompt (str, optional): The user prompt for the instruction. Defaults to "USER: <image>\n".
        assistant_prompt (str, optional): The assistant prompt for the instruction. Defaults to "ASSISTANT:".
        max_new_token (int, optional): The maximum number of new tokens to generate. Defaults to 200.
        process_batch_num (int, optional): Limit the number of batches to process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the processor.

    Returns:
        list: A list of dictionaries containing the generated output, instruction, answer, and other optional fields.
    """
    ret = []
    for batch_i, batch in enumerate(data_loader):
        # print('batch:',batch)
        if process_batch_num:
            if batch_i >= process_batch_num:
                break
        prompt = [f"{user_prompt}{instr['question']}\n{assistant_prompt}" for instr in batch["instr"]]
        img = [asarray(Image.open(img_path).convert('RGB')).transpose(2, 0, 1) for img_path in batch["img_path"]]
        # print("Prompts:", prompt)
        inputs = processor(img, prompt, padding=True, return_tensors="pt", **kwargs).to(device)
        # print("Input IDs:", inputs['input_ids'])
        outputs = model.generate(**inputs, max_new_tokens=max_new_token, do_sample=False, use_cache=True, **kwargs)
        # print("Outputs:", outputs)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
        # print("Generated Text:", generated_text)
        for i, text in enumerate(generated_text):
            # print("Processing text:", text)
            ans = {}
            ans["output"] = text.split(assistant_prompt)[-1]
            ans["instr"] = batch["instr"][i]['question']
            ans["answer"] = batch["instr"][i]["answer"]
            if "id" in batch:
                ans["id"] = batch["id"][i]
            if "idx" in batch:
                ans["idx"] = batch["idx"][i]
            if "img_path" in batch:
                ans["img_path"] = batch["img_path"][i]
            if "img" in batch:
                ans["img"] = batch["img"][i]
            if "bbx_dict" in batch:
                ans["bbx_dict"] = batch["bbx_dict"][i]
            if "labels" in batch:
                ans["labels"] = batch["labels"][i]
            if "boxes" in batch:
                ans["boxes"] = batch["boxes"][i]
            ret.append(ans)
    return ret


def remap_lora_keys(lora_weights):
    new_weights = {}
    for key, value in lora_weights.items():
        if key.startswith('base_model.model.model.vision_tower.vision_tower.'):
            new_key = key.replace('base_model.model.model.vision_tower.vision_tower.', 'base_model.model.vision_tower.')
            new_weights[new_key] = value
        elif key.startswith('model.image_newline'):
            continue
        elif key.startswith('base_model.model.lm_head.weight'):
            new_key = 'base_model.model.language_model.lm_head.weight'
            new_weights[new_key] = value
        elif key.startswith('base_model.model.model.'):
            new_key = key.replace('base_model.model.model.', 'base_model.model.language_model.model.')
            new_weights[new_key] = value
        else:
            new_weights[key] = value
    return new_weights


def remap_keys_nonlora(old_key):

    if old_key.startswith('base_model.'):
        old_key = old_key[11:]

    old_key = old_key.replace('model.model.mm_projector.', 'multi_modal_projector.linear_')
    old_key = old_key.replace('linear_0', 'linear_1')
    old_key = old_key.replace('linear_2', 'linear_2')
    
    return old_key


def load_llava_with_lora(model_id_or_path, checkpoint_lora_path):
    config_path_lora = os.path.join(checkpoint_lora_path, 'adapter_config.json')

    with open(config_path_lora, 'r') as file:
        config = json.load(file)
    config['base_model_name_or_path'] = model_id_or_path
    with open(config_path_lora, 'w') as file:
        json.dump(config, file, indent=4)
    print("Configuration updated successfully!")

    model = LlavaForConditionalGeneration.from_pretrained(model_id_or_path)
    # print(model)
    processor = LlavaProcessor.from_pretrained(model_id_or_path)
    # print(processor)
    processor.patch_size = 16 
    processor.vision_feature_select_strategy = "default"

    non_lora_path = os.path.join(checkpoint_lora_path, 'non_lora_trainables.bin')
    print(non_lora_path)
    non_lora_weights = torch.load(non_lora_path)
    print('Non-LoRA weights loaded.')
    non_lora_weights = {remap_keys_nonlora(k): v for k, v in non_lora_weights.items()}
    model.load_state_dict(non_lora_weights, strict=False)
    print('Non-LoRA weights applied to the model.')

    
    lora_path = os.path.join(checkpoint_lora_path, 'adapter_model.safetensors')

    lora_weights = safetensors.torch.load_file(lora_path)

    remapped_lora_weights = remap_lora_keys(lora_weights)
    model.load_state_dict(remapped_lora_weights, strict=False)
    safetensors.torch.save_file(remapped_lora_weights, lora_path)

    print(f"Remapped weights have been saved to {lora_path}.")

    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, checkpoint_lora_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    return model, processor

def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = [item[key] for item in batch]

    return collated_batch
