import pydicom
from PIL import Image
import numpy as np
import os
import json  # For saving the dictionary as a JSON file
from pydicom.pixel_data_handlers import gdcm_handler, pylibjpeg_handler
import pandas as pd 

def custom_collate_fn(batch):
    # Initialize a dictionary to hold the collated data
    collated_batch = {}

    # Assuming all items in the batch have the same structure
    # Loop over the keys in the first item of the batch to get all attributes
    for key in batch[0]:
        # Collect data for each attribute across all items in the batch
        collated_batch[key] = [item[key] for item in batch]

    return collated_batch


def text_extraction_transform_mimic(text):
    """
    Static method to extract a portion of text based on specific keywords and removes line breaks.
    - If "FINDINGS:" or "REPORT:" is found, keeps only the text after this word until "IMPRESSIONS:" or "CONCLUSIONS:".
    - If neither "IMPRESSIONS:" nor "CONCLUSIONS:" is found after "FINDINGS:" or "REPORT:", keeps the text until the end.
    - If "FINDINGS:" or "REPORT:" cannot be found, only takes what is after "IMPRESSIONS:" or "CONCLUSIONS:" until the end.
    - If none of the keywords are found, keeps the original text.
    - Removes all line break signs from the extracted or original text.
    """
    start_keywords = ["FINDINGS:", "REPORT:", "FINDINGS"]
    end_keywords = ["IMPRESSION:", "CONCLUSIONS:", "IMPRESSION"]
    extracted_text = text  # Default to original text if conditions below do not apply

    # Try to find start keywords first
    start_index = None
    for start_keyword in start_keywords:
        start_index = text.find(start_keyword)
        if start_index != -1:
            start_index += len(start_keyword)
            break

    # If start keyword is found, try to find end keywords
    if start_index is not None:
        end_index = None
        for end_keyword in end_keywords:
            temp_end_index = text.find(end_keyword, start_index)
            if temp_end_index != -1:
                end_index = temp_end_index
                break

        # Extract text between start keyword and end keyword, or until the end if no end keyword is found
        if end_index is not None:
            extracted_text = text[start_index:end_index].strip()
        else:
            extracted_text = text[start_index:].strip()
    if start_index == -1:
        # If no start keyword is found, look for end keywords to start from
        for end_keyword in end_keywords:
            end_index = text.find(end_keyword)
            if end_index != -1:
                start_index = end_index + len(end_keyword)
                extracted_text = text[start_index:].strip()
                break

    return extracted_text.replace("\n", "").replace("\r", "")




def dcm2jpg_resolutions_vindrcxr(data_dir, split, image_size=512):
    """
    Convert DICOM images in a specified subdirectory of data_dir to JPEG format, 
    saving the output in a similarly named subdirectory with "_jpg" appended.
    Also, save the resolutions of the images in a JSON file.
    
    Args:
    data_dir (str): The base directory of the dataset.
    split (str): The subdirectory within data_dir that contains the DICOM files.
    """
    
    input_dir = os.path.join(data_dir, split)
    output_dir = os.path.join(data_dir, split + "_jpg")
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to hold image IDs and resolutions
    resolution_dict = {}
    
    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.lower().endswith(".dicom"):
            dicom_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".jpg")
            
            # Check if JPEG file already exists, if so, skip
            if os.path.exists(output_path):
                continue
            
            try:
                # Read the DICOM file
                ds = pydicom.dcmread(dicom_path)
                
                # Store the resolution in the dictionary as a list [Rows, Columns]
                image_id = os.path.splitext(os.path.basename(dicom_path))[0]
                resolution_dict[image_id] = [ds.Rows, ds.Columns]
                
                # Get the pixel array from the DICOM dataset
                image = ds.pixel_array
                
                # Normalize and convert to 8-bit if necessary
                if image.dtype == np.uint16:
                    image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255.0
                    image = image.astype(np.uint8)
                
                # Convert to PIL image
                im = Image.fromarray(image)
                
                # Resize the image if necessary
                im_resized = im.resize((image_size, image_size), Image.Resampling.LANCZOS)
                
                # Save the resized image as JPEG
                im_resized.save(output_path)
            except Exception as e:
                print(f"Error processing {dicom_path}: {e}")
    
            # Print progress
            print(f"Processed {idx + 1}/{len(os.listdir(input_dir))}: {filename}")
    
    # Save the dictionary to a JSON file without indentation for compactness
    # annotations_dir = os.path.join(data_dir, 'annotations')
    # os.makedirs(annotations_dir, exist_ok=True)  # Ensure the annotations directory exists
    with open(os.path.join(data_dir, 'image_resolutions_' + split + '.json'), 'w') as f:
        json.dump(resolution_dict, f)
    
    print("Conversion and resolution extraction complete.")

# # Example usage:
# data_dir = "/cluster/dataset/medinfmk/ARGON/VinDr-PCXR/"
# split = "test"
# dcm2jpg_resolutions_vindrcxr(data_dir, split)




def find_missing_dicom_ids(splits_path, scene_graph_path):
  """
  This function finds dicom_ids present in splits directory but not scene_graph directory.

  Args:
      splits_path: Path to the directory containing train.csv, valid.csv and test.csv
      scene_graph_path: Path to the directory containing scene graph json files

  Returns:
      list: List of dicom_ids missing in scene_graph directory
  """
  missing_ids = []
  for filename in os.listdir(splits_path):
    if filename.endswith(".csv"):
      csv_path = os.path.join(splits_path, filename)
      data = pd.read_csv(csv_path)
      for dicom_id in data['dicom_id']:
        scene_graph_file = os.path.join(scene_graph_path, f"{dicom_id}_SceneGraph.json")
        if not os.path.isfile(scene_graph_file):
          missing_ids.append(dicom_id)
  return missing_ids