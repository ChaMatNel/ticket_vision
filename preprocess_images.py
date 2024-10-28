# Import libraries
import cv2
import numpy as np
import os

# Define functions for image pre-processing

# Define function to remove transparent pixels in png
def remove_transparent_border(image):
    # Check to see if there is an alpha dimension
    if image.shape[2] != 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    # Find all pixels where the alpha channel is greater than 0 (i.e., non-transparent)
    alpha_channel = image[:, :, 3]
    mask = alpha_channel > 0

    # Get the bounding box of non-transparent pixels
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return image  # if the entire image is transparent, return as is
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the image using the bounding box
    cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]

    return cropped_image

def resize_to_square(image, size=800):
    # Get current height and width of the image
    height, width = image.shape[:2]

    # Compute the padding needed to make the image square
    if height > width:
        padding = (height - width) // 2
        padded_image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    else:
        padding = (width - height) // 2
        padded_image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    # Resize to the target square size
    resized_image = cv2.resize(padded_image, (size, size), interpolation=cv2.INTER_AREA)

    return resized_image

def process_images(input_folder, output_folder, size=800):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all PNG files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Load image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            if image is None:
                print(f"Could not load image {filename}")
                continue

            # Remove transparent border
            cropped_image = remove_transparent_border(image)

            # Resize to square
            square_image = resize_to_square(cropped_image, size)

            # Save the result
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, square_image)

            print(f"Processed and saved: {output_path}")

# Define folder paths
input_folder = r"C:\Users\chada\OneDrive\Desktop\ticket_vision\data\images\holder"
output_folder = r"C:\Users\chada\OneDrive\Desktop\ticket_vision\data\images\holder"

# Run the function
process_images(input_folder, output_folder, size=800)