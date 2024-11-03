import os
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
from PIL import Image
import torch
import pytesseract
from openpyxl import load_workbook

#import functions
from preprocess_images import process_images
from run_model import run_model
from run_ocr import run_ocr
from transform_game_dataframe import transform_game_dataframe
from perform_quality_check import quality_check


# Define folder path
main_folder = r"C:\Users\chada\OneDrive\Desktop\ticket_vision\Games"

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\chada\OneDrive\Desktop\ticket_vision\models\yolov5\runs\train\model_b1_e105\weights\best.pt')

# Path to Tesseract executable (set your path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\chada\OneDrive\Desktop\ticket_vision\models\pytesseract\tesseract.exe'

# Loop through each game file and run the process_images function on all unprocessed images
for game in os.listdir(main_folder):
    game_path = os.path.join(main_folder, game) # Create folder path variable to the game
    if os.path.isdir(game_path):
        
        # Check for Excel files in the specified directory
        excel_files = [f for f in os.listdir(game_path) if f.endswith(('.xlsx', '.xls'))]

        if excel_files:
            # If there are Excel files, load the first one
            file_to_load = os.path.join(game_path, excel_files[0])
            game_df = pd.read_excel(file_to_load)  # Load the Excel file into a DataFrame
        else:
            game_df = pd.DataFrame(columns=["file_path", "object_name", "x_min", "y_min", "x_max", "y_max", "confidence", "img_height", "img_width", "price", "seat_location", "distance_to_center", "deal_score"])
        
        #process image
        process_images(game_path, game_path, size=800)

        for image in os.listdir(game_path):
            if os.path.join(game_path, image).endswith('.png'):
                image_path = os.path.join(game_path, image)
                print(image_path)
                detections_df = run_model(image_path)
                detections_prices_df = run_ocr(detections_df)
                snapshot_df = transform_game_dataframe(detections_prices_df)

                final_snapshot_df = quality_check(snapshot_df)

                game_df = pd.concat([game_df, final_snapshot_df], ignore_index=True)

                processed_path = os.path.join(game_path,'processed')
                os.makedirs(processed_path, exist_ok=True)
                processed_image_path = os.path.join(processed_path, image)
                os.rename(image_path, processed_image_path) 



        # Define output file path
        file_path = f'{game_path}\{game}.xlsx'

        # Check if the file exists to append
        try:
            # Load the existing workbook and append
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                game_df.to_excel(writer, sheet_name=image, index=False)
        except FileNotFoundError:
            # If the file does not exist, create a new file
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                game_df.to_excel(writer, sheet_name=image, index=False)