"""
Step 2: Shows the segmented mask images on top of the original image recorded by the B-ICI.
The user has the option to accept, correct or skip the image. 
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path


# Folder paths
original_folder = Path("/amt-2025-2818/Segmentation_model/<ORIGINAL FOLDER>")
contour_folder = Path("/amt-2025-2818/Segmentation_model/step1_output/<FOLDER NAME>/mask")
output_folder = Path("/amt-2025-2818/Segmentatuin_model/step2_output/<FOLDER NAME>/")

## Get image lists (assuming they have the same names)
# read in the model name that is an extra suffix on the contour images
original_images = sorted(os.listdir(original_folder))
contour_images = sorted(os.listdir(contour_folder))

#  Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


# delete the extra files in case it was deleted already by another run over the data
check_original = {f for f in original_folder.iterdir() if f.is_file()}
check_contour = {f for f in contour_folder.iterdir() if f.is_file()}

normalized_contours = {
        "_".join(p.stem.split("_")[:-2]) + p.suffix
        for p in check_contour
        }

normalized_originals = {p.name for p in check_original}

extra_files = normalized_contours - normalized_originals
print(f"{contour_images=}")
if extra_files:
    for files in extra_files:
        file_path = os.path.join(contour_folder, files)
        os.remove(file_path)
        print(f"Removed file {file_path}")
else:
    print("No extra files to delete")

# Mouse drawing variables
#drawing = False
#draw_color = (0, 0, 0)  # Default color (black)
#brush_size = 2  # Default size for drawing
polygon_points = []
draw_color = (255, 255, 255) #initial color

# Mouse callback function
def draw_contours(event, x, y, flags, param):
    global polygon_points, drawing, draw_color, contour
 
    if event == cv2.EVENT_LBUTTONDOWN:  # Right-click to add a point
        polygon_points.append((x, y))
        drawing = True

        # Draw a small circle at each point
        cv2.circle(contour, (x, y), 1, (0, 0, 0), -1)

        # Draw lines connecting the points
        if len(polygon_points) > 1:
            cv2.line(contour, polygon_points[-2], polygon_points[-1], draw_color, 1)

        # Close and fill polygon if right-clicking near the first point
        if len(polygon_points) > 2 and np.linalg.norm(np.array(polygon_points[0]) - np.array((x, y))) < 10:
            cv2.fillPoly(contour, [np.array(polygon_points, np.int32)], draw_color)
            for px, py in polygon_points: # Change the previous polygon points to white
                cv2.circle(contour, (px, py), 1, draw_color, -1)
            polygon_points.clear()  # Reset the list after filling
            drawing = False  # Stop drawing

def toggle_color(key):
    global draw_color
    if key == ord('p'):
        draw_color = (0, 0, 0) if draw_color == (255, 255, 255) else (255, 255, 255)
        print("Drawing color changed to Black" if draw_color == (0,0,0) else "Color changed to White")

# Loop through all images
for orig_name, contour_name in zip(original_images, contour_images):
    # Load images
    original = cv2.imread(os.path.join(original_folder, orig_name))
    contour = cv2.imread(os.path.join(contour_folder, contour_name))
    original_path = os.path.join(original_folder, orig_name)
    contour_path = os.path.join(contour_folder, contour_name)

    # Ensure images have the same dimensions
    if original.shape != contour.shape:
        print(f"Skipping {orig_name}: Size mismatch")
        continue

    # Create a window and set mouse callback
    cv2.namedWindow("Original image")
    cv2.namedWindow("Edit Contour")
    cv2.setMouseCallback("Edit Contour", draw_contours)
 
    while True:
        # Convert grayscale to colour
        contour_display = contour.copy()
        black_pixels = (contour[:, :, 0] == 0) & (contour[:, :, 1] == 0) & (contour[:, :, 2] == 0) 
        contour_display[black_pixels] = [175, 0, 75]
 
        # Blend images for display
        display = cv2.addWeighted(original, 0.5, contour_display, 0.5, 0)
        cv2.imshow("Original image", original)
        cv2.imshow("Edit Contour", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save and move to next image
#            cv2.imwrite(os.path.join(output_folder, "contour", contour_name), contour)
            shutil.move(contour_path, os.path.join(output_folder, "cropped_contours", contour_name))
#            shutil.move(original_path, os.path.join(output_folder, "cropped_original_png", orig_name))
            polygon_points.clear()
            print(f"Acceptable image, moving {orig_name} to folder")
            break

        elif key == ord('q'):  # Quit
            exit()

        elif key == ord("p"): # Change the drawing color
            toggle_color(key)

        elif key == ord("d"): # Pass
            if os.path.exists(original_path):
#                os.remove(contour_path)
                polygon_points.clear()
                print(f"Passing {contour_path}")
                break

        elif key == ord("r"):  # Crops from the RIGHT
            num_str = input("Enter number of pixels to mask from RIGHT: ")  # Get user input for pixel height
            if num_str.isdigit():  # Ensure it's a valid number
                mask_height = int(num_str)
                if mask_height > 0:
                # Apply white mask to the top `mask_height` pixels
                    contour[:, -mask_height:] = 255  # White mask on top N pixels
                # Overwrite the image
                    cv2.imwrite(contour_path, contour)
                    print(f"Applied and overwrote {contour_path}")
                # Update display with the modified contour
                    display = cv2.addWeighted(original, 0.5, contour, 0.5, 0)
                    cv2.imshow("Edit Contour", display)
                    print(f"Applied white mask of {mask_height} pixels from RIGHT")
            else:
                print("Invalid input, please enter a number.")

        elif key == ord("l"):  # Crops from the LEFT
            num_str = input("Enter number of pixels to mask from LEFT: ")  # Get user input for pixel height
            if num_str.isdigit():  # Ensure it's a valid number
                mask_height = int(num_str)
                if mask_height > 0:
                # Apply white mask to the top `mask_height` pixels
                    contour[:, :mask_height] = 255  # White mask on top N pixels
                # Overwrite the image
                    cv2.imwrite(contour_path, contour)
                    print(f"Applied and overwrote {contour_path}")
                # Update display with the modified contour
                    display = cv2.addWeighted(original, 0.5, contour, 0.5, 0)
                    cv2.imshow("Edit Contour", display)
                    print(f"Applied white mask of {mask_height} pixels from LEFT")
            else:
                print("Invalid input, please enter a number.")

        elif key == ord("t"):  # CCrops from the TOP
            num_str = input("Enter number of pixels to mask from TOP: ")  # Get user input for pixel height
            if num_str.isdigit():  # Ensure it's a valid number
                mask_height = int(num_str)
                if mask_height > 0:
                # Apply white mask to the top `mask_height` pixels
                    contour[:mask_height, :] = 255  # White mask on top N pixels
                # Overwrite the image
                    cv2.imwrite(contour_path, contour)
                    print(f"Applied and overwrote {contour_path}")
                # Update display with the modified contour
                    display = cv2.addWeighted(original, 0.5, contour, 0.5, 0)
                    cv2.imshow("Edit Contour", display)
                    print(f"Applied white mask of {mask_height} pixels from TOP")
            else:
                print("Invalid input, please enter a number.")

        elif key == ord("b"):  # Crops from the BOTTOM
            num_str = input("Enter number of pixels to mask from BOTTOM: ")  # Get user input for pixel height
            if num_str.isdigit():  # Ensure it's a valid number
                mask_height = int(num_str)
                if mask_height > 0:
                # Apply white mask to the top `mask_height` pixels
                    contour[-mask_height:, :] = 255  # White mask on top N pixels
                # Overwrite the image
                    cv2.imwrite(contour_path, contour)
                    print(f"Applied and overwrote {contour_path}")
                # Update display with the modified contour
                    display = cv2.addWeighted(original, 0.5, contour, 0.5, 0)
                    cv2.imshow("Edit Contour", display)
                    print(f"Applied white mask of {mask_height} pixels from BOTTOM")
            else:
                print("Invalid input, please enter a number.")
