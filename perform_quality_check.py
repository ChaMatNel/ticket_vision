import pandas as pd
import cv2

def quality_check(dataframe):
    for index, row in dataframe.iterrows():
        if row['confidence'] <= .8 or row['price'] < 10 or row['price'] > 2000 or row['ocr_conf'] <= .7:
            # Load the image using cv2
            img_check = cv2.imread(row["file_path"])

            if img_check is None:
                print(f"Error loading image at {row['file_path']}")
                continue

            x_min, y_min, x_max, y_max = int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])
            cut_snippet = img_check[y_min:y_max, x_min:x_max]  # Crop the snippet

            # Display the snippet
            cv2.imshow("Image Snippet", cut_snippet)
            cv2.waitKey(0)
            print(f"confirm the price {row['price']} is accurate")
            
            while True:
                try:
                    # Prompt the user to keep the original price or enter a new one
                    user_input = input("Enter a corrected price, or type 'y' to keep the current price: ")
                    
                    if user_input.lower() == 'y':
                        # Keep the existing price
                        new_price = row['price']
                    else:
                        # Convert user input to an integer (new price)
                        new_price = int(user_input)
                    
                    # If the input was valid, break out of the loop
                    break
                except ValueError:
                    print('Invalid input, please enter a number or "y" to keep the current price.')

            # Update the DataFrame with the new price
            dataframe.at[index, 'price'] = new_price
            cv2.destroyAllWindows()
    return dataframe
