import torch
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import pytesseract

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\chada\OneDrive\Desktop\ticket_vision\models\yolov5\runs\train\model_b1_e105\weights\best.pt')

def run_model(folder_path):
    all_detections = []  # Initialize an empty list for each function call
    
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):  # Check if the file is a PNG
            img_path = os.path.join(folder_path, filename)
            print("Processing:", img_path)
            
            # Perform inference on the image
            results = model(img_path)

            # Extract results into a pandas DataFrame
            boxes = results.pandas().xyxy[0]  # Get bounding box results

            # create timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Add image path and required details for each detected object
            for i, row in boxes.iterrows():
                detection_data = {
                    "file_path": img_path,
                    "timestamp": timestamp,  # Add timestamp here
                    "object_name": row["name"],
                    "x_min": row["xmin"],
                    "y_min": row["ymin"],
                    "x_max": row["xmax"],
                    "y_max": row["ymax"],
                    "confidence": row["confidence"]
                }
                all_detections.append(detection_data)
    
    # Convert the list of detections to a DataFrame
    df_detections = pd.DataFrame(all_detections)
    return df_detections

# Run the model on the folder and get the DataFrame
df_detections = run_model(r'C:\Users\chada\OneDrive\Desktop\ticket_vision\data\images\holder')

# Path to Tesseract executable (set your path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\chada\OneDrive\Desktop\ticket_vision\models\pytesseract\tesseract.exe'

# Add a new column for OCR text
df_detections["price"] = ""

# Define the snippets folder path
snippets_folder = r'C:\Users\chada\OneDrive\Desktop\ticket_vision\snippets'

# Create the snippets folder if it doesn't exist
os.makedirs(snippets_folder, exist_ok=True)

for index, row in df_detections.iterrows():
    # Open the image file
    img = Image.open(row["file_path"])

    # Crop the bounding box from the image
    x_min, y_min, x_max, y_max = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
    cropped_img = img.crop((x_min, y_min, x_max, y_max))

    # Calculate new width and crop to remove left 3/8 of the snippet
    width, height = cropped_img.size
    new_x_min = int(width * 3/8)
    cropped_img = cropped_img.crop((new_x_min, 0, width, height))

    # Convert the cropped image to grayscale for better OCR accuracy
    cropped_img = cropped_img.convert('L')
    
    # Save the cropped image in the snippets folder
    snippet_path = os.path.join(snippets_folder, f'snippet_{index}.png')
    cropped_img.save(snippet_path)

    # Run Tesseract OCR on the cropped image
    text = pytesseract.image_to_string(cropped_img, config='--psm 6')  # Adjust `psm` mode as needed for best results
    
    # Add the OCR result to the DataFrame
    df_detections.at[index, "price"] = text.strip()

# Check the DataFrame with extracted text
print(df_detections)
df_detections.to_excel(r'C:\Users\chada\OneDrive\Desktop\ticket_vision\test4.xlsx')