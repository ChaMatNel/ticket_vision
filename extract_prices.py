# Import packages
import os
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
from PIL import Image
import torch
import pytesseract
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

# Import functions
from preprocess_images import process_images
from run_model import run_model
from run_ocr import run_ocr
from transform_game_dataframe import transform_game_dataframe
from perform_quality_check import quality_check
from create_trend_graph import create_trend_graph


# Define folder path
main_folder = r"C:\Users\chada\OneDrive\Desktop\ticket_vision\test_games"

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\chada\OneDrive\Desktop\ticket_vision\models\yolov5\runs\train\model_b1_e105\weights\best.pt')

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\chada\OneDrive\Desktop\ticket_vision\models\pytesseract\tesseract.exe'

# Loop through each game file and run the process_images function on all unprocessed images
for game in os.listdir(main_folder):
    game_path = os.path.join(main_folder, game) # Create folder path variable to the game
    if os.path.isdir(game_path):
        
        # Check for Excel files in the specified directory
        excel_files = [f for f in os.listdir(game_path) if f.endswith(('.xlsx', '.xls'))]

        if excel_files:
            # If there are Excel files, load the first one and create a pandas dataframe
            file_to_load = os.path.join(game_path, excel_files[0])
            game_df = pd.read_excel(file_to_load)

        else:
            # If there isn't an Excel file, create a new pandas dataframe that will later be exported as an excel file
            game_df = pd.DataFrame(columns=["file_path", "object_name", "x_min", "y_min", "x_max", "y_max", "confidence", "img_height", "img_width","ocr_conf", "price", "seat_location", "distance_to_center", "deal_score", "snapshot_date", "days_to_game"])
        
        # Run the process_images function which resizes the image to a square
        process_images(game_path, game_path, size=800)

        # Loop through each file/image in the game folder
        for image in os.listdir(game_path):
            # If the file is an image, run various functions to extract the ticket prices and put the information in a dataframe
            if os.path.join(game_path, image).endswith('.png'):
                image_path = os.path.join(game_path, image)
                # Run the YOLOv5 model to detect price tag bounding boxes
                detections_df = run_model(image_path)
                # Run OCR on the ticket price tag snippets to extract the price
                detections_prices_df = run_ocr(detections_df)

                #clean price column
                detections_prices_df['price'] = detections_prices_df['price'].str.extract('(\d+)')  # Extract only digits
                detections_prices_df['price'] = detections_prices_df['price'].astype(float)  # Convert to float first if decimals are present
                detections_prices_df['price'] = detections_prices_df['price'].fillna(0).astype(int)

                # Run the transform_game_dataframe function which creates new calculated columns for later analysis
                snapshot_df = quality_check(detections_prices_df)
                # Run though all the snippets and manually check low confidence snippets
                final_snapshot_df = transform_game_dataframe(snapshot_df)
                # Add the game information to the main dataframe 
                game_df = pd.concat([game_df, final_snapshot_df], ignore_index=True)
                #need to do quality check before some of the data tansformations so deal score is accurate
                
                # Move the images to a "processed" folder
                processed_path = os.path.join(game_path,'processed')
                os.makedirs(processed_path, exist_ok=True)
                processed_image_path = os.path.join(processed_path, image)
                os.rename(image_path, processed_image_path)

        # Define output file path
        file_path = rf'{game_path}\{game}.xlsx'
        png_path = rf'{game_path}\{game}.png'

        #create_trend_graph(game_df, png_path)

        # Export the dataframe as an excel file
        try:
            # Load the existing workbook and append
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                game_df.to_excel(writer, sheet_name=image, index=False)
        except FileNotFoundError:
            # If the file does not exist, create a new file
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                game_df.to_excel(writer, sheet_name=image, index=False)