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
        
        #process image
        process_images(game_path, game_path, size=800)

        for image in os.listdir(game_path):
            if os.path.join(game_path, image).endswith('.png'):
                image_path = os.path.join(game_path, image)
                print(image_path)
                df_detections = run_model(image_path)
                game_details = run_ocr(df_detections)
                print(game_details)
                final_game_details = transform_game_dataframe(game_details)

                # Define output file path
                file_path = f'{game_path}\{game}.xlsx'

                # Check if the file exists to append
                try:
                    # Load the existing workbook and append
                    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                        final_game_details.to_excel(writer, sheet_name='Sheet1', index=False)
                except FileNotFoundError:
                    # If the file does not exist, create a new file
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        final_game_details.to_excel(writer, sheet_name='Sheet1', index=False)