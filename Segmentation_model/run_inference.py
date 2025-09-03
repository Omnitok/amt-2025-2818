"""""""""
This script is created to evaluate the converted "*.png" images from the B-ICI instrument.
Creates a txt file with the output of:
name of the particle
AR - aspect ratio
particle size (in pixels)
AED - Area Equivalent Diameter
min_dia
max_dia
min_circle_diameter / maximum_dimension
x - length
y - width
border - Particle is on border(1) or not(0)?
"""""""""
import onnxruntime as ort
import sys
from typing import List
import argparse
from pathlib import Path
import cv2
import os
import tqdm
import numpy as np
from pathlib import Path
import math
import model.dataset as dataset
import shutil

model_size = (
    960,
    1280,
)

def _merge_images(orignal, add):
    org_shape = orignal.shape
    add_shape = add.shape

    if add_shape[0] >= org_shape[0] and add_shape[1] >= org_shape[1]:
        orignal = add[: org_shape[0], : org_shape[1]]
    elif add_shape[0] <= org_shape[0] and add_shape[1] <= org_shape[1]:
        orignal[: add_shape[0], : add_shape[1]] = add
    else:
        assert (
            False
        ), f"Inconsistent shapes, cannot crop or fill {add_shape} vs {model_size}"
    return orignal

def calculate_max_dim(binary_image_path): # diameter of the smallest circle that encloses the particle 
     # Read the binary image
     binary_image = binary_image_path 
 
     # Find contours
     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
     # Fit a circle to the contours
     (x, y), radius = cv2.minEnclosingCircle(contours[0])
     center = (int(x), int(y))
     radius = int(radius)

     output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
 
     # Calculate the diameter times 1.65 um/px
     diameter = radius * 2 * 1.65
 
  # Define the top-left and bottom-right coordinates of the bounding box
     top_left = (max(0, int(x - radius * 1.1)), max(0, int(y - radius * 1.1)))
     bottom_right = (min(output_image.shape[1], int(x + radius * 1.1)), min(output_image.shape[0], int(    y + radius * 1.1)))
 
     # Extract the circle area from the image
     circle_area = output_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
 
     # Return the diameter
     return diameter  
   
def main(args):
    model = ort.InferenceSession(
        str(args.model),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    # CREATE FOLDER IF ID DOESNT EXIST
    folder_names = ["particles", "mask", "bbox", "overlay", "mask_particle", "contoured_particle", "circle"]

    if os.path.exists(args.output_folder): 
        shutil.rmtree(args.output_folder)
        print("\n Folder exists. Deleting...")
    os.mkdir(args.output_folder)
    print(f"\n Creating folder: {args.output_folder}")
    
    for folder_name in folder_names:
        folder_path = os.path.join(args.output_folder, folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Creating folder: {folder_name}")

    ### CREATE TEXTFILE
    with open(str(args.output_folder) + "/particles.txt", "w") as file:
        file.write(
        "Particle_ID " + 
        "aspect_ratio " + 
        "aed " + 
        "min_dia " +
        "max_dim " + 
        "x " +
        "y " +
        "border " +
        "area_ratio " +
        "particle_area " +
        "particle_area_2 " +
        "\n")
    
        for image_files in tqdm.tqdm(os.listdir(args.input_folder)):
            img_path = (os.path.join(args.input_folder, image_files))
            img_name = Path(img_path).stem
            if img_path.endswith(".png"):
                img_u8c = cv2.imread(img_path)
                if img_u8c.shape[-1] == 3:
                    img_u8 = cv2.cvtColor(img_u8c, cv2.COLOR_BGR2GRAY)

                norm_value = (
                    np.iinfo(np.uint16).max
                    if np.max(img_u8) > np.iinfo(np.uint8).max
                    else np.iinfo(np.uint8).max
                )

                img_fp = img_u8.astype(dtype=np.float32) / norm_value
                input = np.zeros((1, 960, 1280, 1), dtype=np.float32)

                input[0, :, :, 0] = _merge_images(input[0, :, :, 0], img_fp)
                output = model.run(None, {"image_input": input})[0][0]
                
                mask_model_size = np.array(255 * (output[:, :, 0] >= 0.99), dtype=np.uint8)
                
                mask = np.zeros(img_fp.shape)
                mask = _merge_images(mask, mask_model_size)
                mask = cv2.bitwise_not(mask) # Invert B/W
                cv2.imwrite(
                        str(Path(args.output_folder) /
                            "mask" /
                            f"mask_{image_files}"),
                        mask
                        )

                ### FIND CONTOURS/BOUNDING BOX
                edged = cv2.Canny(mask, 1, 254)
                contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                filtered_contours = []
                for contour_box in contours:
                    rect = cv2.minAreaRect(contour_box)
                    _, size, _ = rect
                    w, h = map(int, size)

                    if w > 0 and h > 0:
                        filtered_contours.append(contour_box)

                contour_image = img_u8c
                contour = cv2.drawContours(img_u8c, contours, -1, (0,255,0), 1)
                contour2 = contour
                px_res = 1.65
                scale = 1.75
                counter = 1

                for contour_box in contours:
                    # get the bounding box of the contours
                    rect = cv2.minAreaRect(contour_box)
                    center_rot, size_rot, angle_rot = rect
                    w_rot, h_rot = size_rot
                    x_rot, y_rot = center_rot

					# Get the information of the NOT rotated bounding box
                    x, y, w, h = cv2.boundingRect(contour_box)
                    center_x = x + w / 2
                    center_y = y + h / 2
                    center_x_rot = x_rot + w_rot / 2
                    center_y_rot = y_rot + h_rot / 2

					# Fill contour if it touches image edge
                    im_height, im_width = contour_image.shape[:2]
                    for point in contour_box:
                        x, y = point [0]
                        if x <= 0 or x >= (im_width - 1) or y >= 0 or y >= (im_height - 1):
                            contour_box = contour_box.reshape(-1,2)
                            contour_box = np.vstack([contour_box, contour_box[0]])
                            contour = cv2.drawContours(contour_image, [contour_box], -1, (0,255,0), 1)
                            break

                # Check if the bounding box crosses the border
                    if (x - w / 2) < 0 or (x + w / 2) > img_u8.shape[1] or (y - h / 2) < 0 or (y + h / 2) > img_u8.shape[0]:
                        border = 1
                    else:
                        border = 0

                    scaled_w = (w * scale)
                    scaled_h = (h * scale)
                   
                    # positioning the scaled rectangle to encapsulate the particle
                    start_x = max(int(center_x - scaled_w // 2), 0)
                    start_y = max(int(center_y - scaled_h // 2), 0)
                    end_x = min(int(center_x + scaled_w // 2), img_u8.shape[1])
                    end_y = min(int(center_y + scaled_h // 2), img_u8.shape[0])

                    # Draw rectangle around the contour
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    contour_rectangle = cv2.drawContours(contour, [box], 0, (0, 255, 255), 2)

                # EXTRACT THE SCALED REGION INSIDE THE BOUNDING BOX
                    particle = img_u8[ start_y:end_y, start_x:end_x ]
                    particle_mask = mask[ start_y:end_y, start_x:end_x ]
                    particle_contour = contour[ start_y:end_y, start_x:end_x ]
                    s_x = max(int(center_x - w // 2), 0)
                    s_y = max(int(center_y - h // 2), 0)
                    e_x = min(int(center_x - w // 2), mask.shape[1])
                    e_y = min(int(center_y - h // 2), mask.shape[0])
                    particle_mask_nonscaled = mask[ s_y:e_y, s_x:e_x ]

                    particle_area = np.count_nonzero(particle_mask == 255)
                    particle_area_2 = np.count_nonzero(particle_mask_nonscaled == 255)
                    
                    threshold = 10 / px_res * scale
                    if w_rot > threshold and h_rot > threshold :  # LIMIT THE OBJECT SIZE
                        cv2.imwrite(
                            str(Path(args.output_folder) /
                                "particles" /
                                 f"particle_{str(img_name)}_{counter}.png"),
                                 particle
                                 )
                        cv2.imwrite(
                              str(Path(args.output_folder) /
                                 "mask_particle" /
                                 f"particle_mask{str(img_name)}_{counter}.png"),
                                 particle_mask
                                 )
                        cv2.imwrite(
                              str(Path(args.output_folder) /
                                 "contoured_particle" /
                                 f"particle_{str(img_name)}_{counter}.png"),
                                 particle_contour
                                 )
                      ## calculate the aspect ratio of particles that is defined by the ratio of the 2 length of the bounding box that encapsulates the particle.
                        aspect_ratio = max(h_rot, w_rot) / min(h_rot, w_rot)
                        
                        ## calculate the size of the particle
                        # AED - Area Equivalent Diameter converted to um
                        aed = 2 * math.sqrt(particle_area / math.pi) * px_res
 
			# Min diameter is the diameter of the circle calculated from the area
                        # AREA EQUIVALENT DIAMETER DERIVED FROM THE GEOMETRIC MEAN OF THE CIRCLE -- SHOULD BE SIMILAR TO AED
                        min_dia = 2 * math.sqrt((w_rot * h_rot) / math.pi) * px_res

                        # maximum_dimension / ferret
                        _, radius = cv2.minEnclosingCircle(contour_box)
                        max_dim =  (radius) * px_res * 2

                        print(f"BIG ENOUGH: {img_name}_{counter}  \n"
                              f"Area: {particle_area} pixel \n"
                              f"its AED: {aed} \n")
                        # Area ratio
                        area_ratio = particle_area / (math.pi/4 * aed*2)

                        file.write(f"{str(img_name)}_{counter} " +
                                f"{aspect_ratio} " +
                                f"{aed} " +
                                f"{min_dia} " +
                                f"{max_dim} " +
                                f"{x} " +
                                f"{y} "+
                                f"{border} " +
                                f"{area_ratio}" +
                                f"{particle_area}" +
                                f"{particle_area_2}" +
                                "\n" )
                            
                    else:
                        file.write(f"{str(img_name)}_{counter} "+ "particle smaller than 10um"+ "\n")

                        print(f"\n {img_name}_{counter} NOT exported \n"
                              )
                    contour_rectangle = cv2.putText(
                        contour_rectangle,
                        f"{counter}",
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        2,
                        cv2.LINE_AA
                        )

                    counter = counter + 1
                
                    cv2.imwrite(str(Path(args.output_folder) /
                        "bbox" /
                        f"box_{image_files}"),
                        contour_rectangle
                        )

                ### FIND THE SMALLEST CIRCLE/FERRET DIAMETER
                for bounding_circle in contours:
               
                    (circle_x, circle_y), radius = cv2.minEnclosingCircle(bounding_circle)
                    center = (int(circle_x), int(circle_y))
                    radius = int(radius)
                    
                    contour2 = cv2.circle(contour2, center, radius, (0, 0, 255), 2)

                cv2.imwrite(str(Path(args.output_folder) /
                    "circle" /
                    f"circle_{image_files}"),
                    contour2
                    )
                
                img_3ch = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
                overlay = _merge_images(img_3ch[:, :, 2], mask)
                img_3ch[:, :, 2] = np.where(overlay, 255, img_3ch[:, :, 2])
                cv2.imwrite(
                    str(Path(args.output_folder) /
                        "overlay" /
                        f"overlay_{image_files}"),
                        img_3ch
                        )
            
def parse_args(args: List[str]):

    parser = argparse.ArgumentParser(
        "Run model inference", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the dataset folder",
    )

    parser.add_argument(
        "--input_folder",
        type=Path,
        required=True,
        help="Path to read the pngs from",
    )

    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Path to dump the data",
    )

    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
