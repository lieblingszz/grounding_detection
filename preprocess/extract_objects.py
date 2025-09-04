import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd



def extract_info_from_file(file):
    """
    Extract bounding box information from a JSON file, including the image ID.
    Each bounding box's information is returned as a dictionary.
    Args:
        file (str): Path to the JSON file.
    Returns:
        list of dicts: List containing dictionaries with bbox information.
    """
    with open(file, 'r') as f:
        data = json.load(f)
        image_id = data['image_id']
        results = []
        for obj in data['objects']:
            bbox_info = {
                'image_id': image_id,
                'bbox_name': obj['bbox_name'],
                'x1': float((obj['x1']) / 224),
                'y1': float((obj['y1']) / 224) ,
                'x2': float((obj['x2']) / 224), 
                'y2': float((obj['y2']) / 224)
            }
            results.append(bbox_info)
        return results

def process_files_batch(files):
    """
    Processes a batch of files using ThreadPoolExecutor, accumulating results into a list.
    Args:
        files (list): List of file paths to process.
    Returns:
        list of dicts: Aggregated results from all files in the batch.
    """
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(extract_info_from_file, file) for file in files]
        for future in tqdm(as_completed(futures), total=len(files)):
            results.extend(future.result())
    return results

def process_json_files(folder_path, output_file_path):
    """
    Processes all JSON files in a folder in batches and saves the results to a CSV file incrementally.
    """
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('_SceneGraph.json')]
    
    batch_size = 100
    first_batch = True
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        batch_results = process_files_batch(batch_files)
        df = pd.DataFrame(batch_results)
        if first_batch:
            df.to_csv(output_file_path, mode='w', index=False, header=True)
            first_batch = False
        else:
            df.to_csv(output_file_path, mode='a', index=False, header=False)
        print(f"Processed batch {i // batch_size + 1}/{(len(all_files) - 1) // batch_size + 1}")

    print(f"Data saved to {output_file_path}")

def add_image_paths_to_output(output_file, train_file, validation_file, test_file, train_output, valid_output, test_output):
    """
    Adds image paths to the output CSV file by matching image IDs with image paths, and then split the results into train, validation, and test CSVs based on the source of image ID.
    Args:
        output_file (str): Path to the output CSV file.
        train_file (str): Path to the train CSV file containing image IDs and paths.
        validation_file (str): Path to the validation CSV file containing image IDs and paths.
        test_file (str): Path to the test CSV file containing image IDs and paths.
        train_output (str): Path for the output train CSV file.
        valid_output (str): Path for the output validation CSV file.
        test_output (str): Path for the output test CSV file.
    """
    train_df = pd.read_csv(train_file)
    validation_df = pd.read_csv(validation_file)
    test_df = pd.read_csv(test_file)
    output_df = pd.read_csv(output_file)
    
    train_df['source'] = 'train'
    validation_df['source'] = 'validation'
    test_df['source'] = 'test'
    
    all_images_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    
    output_df['image_path'] = output_df['image_id'].map(all_images_df.set_index('dicom_id')['path'].apply(lambda x: x.replace('.dcm', '.jpg')))
    output_df['source'] = output_df['image_id'].map(all_images_df.set_index('dicom_id')['source'])
    output_df[output_df['source'] == 'train'].to_csv(train_output, index=False)
    output_df[output_df['source'] == 'validation'].to_csv(valid_output, index=False)
    output_df[output_df['source'] == 'test'].to_csv(test_output, index=False)

if __name__ == "__main__":
    # Update these paths to your local data directories
    folder_path = './data/CHEST_IMA/silver_dataset/scene_graph'
    output_file_path = 'extracted_objects.csv'
    train_output = 'llava_train_01.csv'
    valid_output = 'llava_valid_01.csv'
    test_output = 'llava_test_01.csv'
    process_json_files(folder_path, output_file_path)
    add_image_paths_to_output(output_file_path, './data/CHEST_IMA/silver_dataset/splits/train.csv', './data/CHEST_IMA/silver_dataset/splits/valid.csv', './data/CHEST_IMA/silver_dataset/splits/test.csv', train_output, valid_output, test_output)
