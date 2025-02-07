import os
from PIL import Image
import pytesseract

def run_ocr(df_detections):

    # Add a new column for the price that will be extracted with OCR
    #df_detections['img_height'] = 0
    #df_detections['img_width'] = 0
    df_detections["ocr_conf"] = ""
    df_detections["price"] = ""

    # Iterate through each bounding box in the detections dataframe
    for index, row in df_detections.iterrows():

        # Load in image based on filepath provided in detections dataframe and set predicted bounding box coordinates as variables
        img = Image.open(row["file_path"])
        x_min, y_min, x_max, y_max = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
        
        # Cut the snippet and process the snippet
        cut_snippet = img.crop((x_min, y_min, x_max, y_max)) # Extract the snippet from the image
        width, height = cut_snippet.size #define height and width of snippet

        #df_detections['img_height'] = height
        #df_detections['img_width'] = width

        new_x_min = int(width * 3/8) # Define new left border of snippet since it doesn't contain the price information
        crop_snippet = cut_snippet.crop((new_x_min + 1, 3, width - 1, height - 3)) # crop the snippet 
        enlarged_coords = (crop_snippet.size[0] * 4, crop_snippet.size[1] * 4)
        enlarged_snippet = crop_snippet.resize(enlarged_coords, Image.LANCZOS) #enlarge the snippet
        
        # Create a clearer number for OCR by converting dark pixels to black and converting lighter pixels to white
        snippet_binarized = enlarged_snippet.convert('L').point(lambda p: 0 if p <= 115 else 255) # 115 seems to be sweetspot

        # Run OCR on the processed snippet
        ocr_data = pytesseract.image_to_data(snippet_binarized, config='--psm 6 -c tessedit_char_whitelist=$0123456789li|', output_type=pytesseract.Output.DICT)
        text = ''.join(ocr_data['text']).replace("l", "1").replace("i", "1").replace("|", "1")
        valid_confidences = [x for x in ocr_data['conf'] if x != -1]
        confidence_score = min(valid_confidences) if valid_confidences else 0

        #text = text.replace(" ", "_").replace("l", "1").replace("i", "1").replace("|", "1").replace("\n", "") # Replace characters that are similar to "1" with "1"

        # Handle cases where OCR fails to detect text
        if text == "":
            text = "99999" # Set to really high value that can be easily identified and manually corrected later
        
        # Add the OCR result to the DataFrame
        df_detections.at[index, "price"] = text.strip()
        # Add the confidence score
        df_detections.at[index, "ocr_conf"] = confidence_score/100
    
         # Restart the loop at the next row
    
    return df_detections