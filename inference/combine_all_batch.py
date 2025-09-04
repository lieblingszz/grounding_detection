import json
import os

# modify this code to specify the directory
batch_files_dir = "./eval_output/batches_all_llava"  # Update to your local output directory  


batch_files = [f"eval_output_batch_{i}_all.json" for i in range(3)]


combined_data = []


for batch_file in batch_files:
    batch_file_path = os.path.join(batch_files_dir, batch_file)

    with open(batch_file_path, "r") as f:
        batch_data = json.load(f)
        

        for entry in batch_data:
            if isinstance(entry, list):
                combined_data.extend(entry)
            else:
                combined_data.append(entry)

# modify this code to specify the name
combined_output_path = os.path.join(batch_files_dir, "eval_all_batches_vindr.json")
with open(combined_output_path, "w") as f:
    json.dump(combined_data, f)

print(f"Combined data saved to {combined_output_path}.")
