import pandas as pd
import cv2

def quality_check(dataframe):
    for index, row in dataframe.iterrows():
        if row['confidence'] <= .8 or row['price'] < 10 or row['price'] > 2000:
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
            
            while True:
                try:
                    new_price = int(input("Enter a corrected price: "))  # Prompt user for new value
                    break
                except ValueError:
                    print('Invalid input')
            dataframe.at[index, 'price'] = new_price  # Overwrite the 'price' column
            cv2.destroyAllWindows()
    return dataframe
