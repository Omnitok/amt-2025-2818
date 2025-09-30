"""""""""
Step 3 crops the segmented images from the B-ICI instrument.
Creates a txt file with the output of:
name of the particle
AR - aspect ratio
particle size (in pixels)
AED - Area Equivalent Diameter - in micrometer
min_dia - in micrometer
max_dia - in micrometer
min_circle_diameter / maximum_dimension - in micrometer
x - length - in pixels
y - width - in pixels
border - Particle is on border(1) or not(0)?
"""""""""
#import sys
#from typing import List
#import argparse 
from pathlib import Path
import cv2
import os
import tqdm
import numpy as np
from pathlib import Path
import math
import shutil

input_folder = Path("/amt-2025-2018/Segmentation_model/<INPUT_FOLDER>")
contour_folder = Path("/amt-2025-2018/Segmentation_model/<CONTOUR_FOLDER>")
output_folder = Path("/amt-2025-2018/Segmentation_model/<OUTPUT_FOLDER>")

debug = 0 
px_res = 1.65
threshold_um = 20
threshold = threshold_um / px_res
scale = 0.75

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

def is_contour_closed(contour, tolerance=1):
    first_point = contour[0,0]
    last_point = contour[-1,0]
    is_closed = np.linalg.norm(first_point - last_point) < tolerance
    return is_closed, first_point, last_point

    # CREATE Output FOLDER IF ID DOESNT EXIST

folder_names = ["particles", "mask", "bbox", "overlay", "mask_particle", "contoured_particle", "circle", "nonscaled_particle"]

if os.path.exists(output_folder): 
    shutil.rmtree(output_folder)
    print("\n Folder exist, deleting...")
os.mkdir(output_folder)
print(f"Creating folder: {output_folder}")
   
for folder_name in folder_names:
    folder_path = os.path.join(output_folder, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print(f"Creating folder: {folder_name}")

# Add the model_name as a suffix
example_file = next(contour_folder.glob("*.png"))
#version = "_" + "_".join(example_file.stem.split("_")[-2:])
version = "" # USE THIS IF THERE IS NO MODEL-VERSION IN THE IMAGE NAMES

### CREATE TEXTFILE
with open(str(output_folder) + "/particles_" + version + ".txt", "w") as file:
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
        "particle_area_3 " +
        "particle_area_0 " +
        "closed? " +
        "image_shape " +
        "\n"
    )

    for image_files in tqdm.tqdm(os.listdir(input_folder)):
        img_path = (os.path.join(input_folder, image_files))
        img_name = Path(img_path).stem
        if debug == 1:
            print(f"Original file: {image_files}\n")
            print(f"{img_path=}\n")       

        if not img_path.endswith(".png"):
            print("...will be skipped")
        else:
            img_name_4contour = Path(image_files)
            img_u8c = cv2.imread(str(img_path))
            image_shape = img_u8c.shape[:2] 
            try:
                #img_u8c.shape[-1] == 3
                img_u8 = cv2.cvtColor(img_u8c, cv2.COLOR_BGR2GRAY)
            except:
                print(f"Countour image shape is wrong: {image_shape}")  
            
            norm_value = (
                np.iinfo(np.uint16).max
                if np.max(img_u8) > np.iinfo(np.uint8).max
                else np.iinfo(np.uint8).max
            )

            img_fp = img_u8.astype(dtype=np.float32) / norm_value
            # Add the model_name as a suffix
            #            example_file = next(contour_folder.glob("*.png"))
            #            version = "_" + example_file.stem.split("_")[-1]

    ###########################CHANGED HERE
            contour_name = f"{img_name_4contour.stem}{version}{img_name_4contour.suffix}"
            #            contour_name = f"mask_{img_name_4contour.stem}{img_name_4contour.suffix}"
            contour_path = os.path.join(contour_folder, contour_name)

            if debug == 1:
                print(f"{contour_path=}\n")
            binary_image = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
            #print(f"{binary_image=}\n")
            
            if binary_image is None:
                print(f"{binary_image} is None")
                continue

    ############################### CHANGED HERER        
            mask_model_size = np.array(255 * (binary_image[:, :] >= 0.99), dtype=np.uint8)
            #mask_model_size = 255 - np.array(255 * (binary_image[:, :] >= 0.99), dtype=np.uint8)
    
            mask = np.zeros(img_fp.shape)
            mask = _merge_images(mask, mask_model_size)
            mask = cv2.bitwise_not(mask) # Invert B/W
            cv2.imwrite(
                    str(Path(output_folder) /
                        "mask" /
                        f"{image_files}"),
                    mask
                    )

            ### FIND CONTOURS/BOUNDING BOX
#            edged = cv2.Canny(mask, 1, 254)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            filtered_contours = []
            for contour_box in contours:
                rect = cv2.minAreaRect(contour_box)
                _, size, _ = rect
                w, h = map(int, size)

                if w > 0 and h > 0:
                    filtered_contours.append(contour_box)

            contour_image = img_u8c.copy()
            contour = cv2.drawContours(contour_image, contours, -1, (0,255,0), 1)
            contour2 = img_u8c.copy()
            counter = 1 # Only particles that bigger than the treshold, total particles on the image
            counter_total = 1 # Any object on the image 
                         
            for contour_box in filtered_contours:
                rect = cv2.minAreaRect(contour_box)
                center_rot, size_rot, angle_rot = rect
                w_rot, h_rot = size_rot
                x_rot, y_rot = center_rot
				
                # If particles are too small, skip and move to the next
                if w_rot < threshold and h_rot < threshold:
                    file.write(f"{img_name}_000 particle smaller than {threshold_um} um"
                            f", minAreaRect: {rect}\n")
                    #print(f"{str(img_name)}_000 NOT exported: minAreaRect: {rect}\n")
                    counter_total = counter_total + 1
                    continue

                # TK DRAW THIS FIRST (then contour): Draw rectangle around the contour
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(contour_image, [box], 0, (255, 0, 0), 1)
                # TK Draw ONLY one contour (that belongs to the particle)in Red
                cv2.drawContours(contour_image, [contour_box], 0, (0,0,255), 1)
                # TK draw again for single particle: contoured_particle
                contour_single = contour_image.copy()
                cv2.drawContours(contour_single, [box], 0, (255, 0, 0), 1)
                cv2.drawContours(contour_single, [contour_box], 0, (0, 0, 255), 1)

                # TK fill contours to count pixels for area
                mask_single = np.zeros(img_u8c.shape, np.uint8)

                # TK thickness= -1 for contour to be drawn filled
                FILLED = -1
                cv2.drawContours(mask_single, [contour_box], 0, (255, 255, 255), FILLED)
                
				# Get the information of the bounding box, not rotated
                x, y, w, h = cv2.boundingRect(contour_box)
                center_x = x + w / 2
                center_y = y + h / 2
                
                # Fill contour if it touches image edge
#                im_height, im_width = contour_image.shape[:2]
#                for point in contour_box:
#                    x_check, y_check = point[0]
#                    if x_check <= 0 or x_check >= (im_width - 1) or y_check <= 0 or y_check >= (im_height - 1):
#                        contour_box = contour_box.reshape(-1,2)
#                        contour_box = np.vstack([contour_box, contour_box[0]])
#                        contour = cv2.drawContours(contour_image, [contour_box], -1, (0,255,0), 1)
#                        break

				# Check if the bounding box crosses the border
                if x == 0 or (x + w) == img_u8.shape[1] or y == 0 or (y + h) == img_u8.shape[0]:
                    border = 1
                else:
                    border = 0

                scaled_w = (w * scale)
                scaled_h = (h * scale)

#                start_x = max(int(center_x - scaled_w // 2), 0)
#                start_y = max(int(center_y - scaled_h // 2), 0)
#                end_x = min(int(center_x + scaled_w // 2), img_u8.shape[1])
#                end_y = min(int(center_y + scaled_h // 2), img_u8.shape[0])
                half_scale = scale / 2
                padding_width = w * half_scale
                padding_height = h * half_scale
                start_x = max(int(x - padding_width), 0)
                start_y = max(int(y - padding_height), 0)
                end_x = min(int(x + (w + padding_width)), img_u8.shape[1])
                end_y = min(int(y + (h + padding_height)), img_u8.shape[0])
                
				# crop the tight bound rectangle to calculate the area of the object
                s_x = max(x, 0)
                s_y = max(y, 0)
                e_x = min(x + w, mask.shape[1])
                e_y = min(y + h, mask.shape[0])
                particle_mask_nonscaled = mask[ s_y:e_y, s_x:e_x ]
                particle_nonscaled = contour[ s_y:e_y, s_x:e_x ]
                xw = x + w
                yh = y + h
                
                # EXTRACT THE SCALED REGION INSIDE THE BOUNDING BOX
                particle = img_u8[ start_y:end_y, start_x:end_x ]
#                particle = contour[ start_y:end_y, start_x:end_x ]
                particle_mask = mask[ start_y:end_y, start_x:end_x ]
                particle_contour = contour_single[ start_y:end_y, start_x:end_x ]

				# crop the tight bound rectangle to calculate the area of the object
                closed = is_contour_closed(contour_box)
                particle_area3 = cv2.contourArea(contour_box) * (px_res ** 2)
                particle_area = np.count_nonzero(particle_mask == 255) * (px_res**2) # um2
                particle_area_0 = np.count_nonzero(particle_mask == 0) * (px_res**2)
                particle_area2 = np.count_nonzero(particle_mask_nonscaled == 255) * (px_res**2) # Fehér: 255 / Fekete: 0 

#################################################### validate the new points
#                if start_x <= end_x or start_y <= end_y:
#                    print(f"Invalid bounding box in {img_name}_{counter}: start_x={start_x}, end_x={end_x}, start_y={start_y}, end_y={end_y}")
#                    while True:
#                        cv2.imshow("eredeti", img_u8)
#                        cv2.imshow("forgatott", contour)
#                        cv2.imshow("mask", particle_mask)
#                        cv2.imshow("nonscaled mask", particle_mask_nonscaled)
#                        key = cv2.waitKey(50)
#                        if key == ord("q"):
#                            break
#                    continue    
########################################################## DEBUG

                if debug == 1:
                    print(f"particle name: {str(img_name)}_{counter_total} \n" 
#                      f"M:  {M} \n"
#                      f"image_dtype: {img_u8c.dtype} \n" 
                      f"particle shape: {particle.shape} \n"
#                      f"shape1: {particle.shape[1]} \n"
                      f"size (w,h): {w}, {h} \n"
                      f"size (w_rot, h_rot): {w_rot}, {h_rot} \n"
                      f"Is the particle closed?: {closed[0]} \n"
                      f"first and last point: {closed[1]} -- {closed[2]} \n"
#                      f"angle: {angle_rot} \n"
#                      f"particle_mask shape: {particle_mask.shape} \n"
                      f"particle area 1/2/3: {particle_area}/{particle_area2}/{particle_area3} \n"
                      f"Object size W: {w}, H: {h}, threshold(either): {threshold} \n"
                      f"s_x/x: {s_x}/{x} \n e_x/x+w: {e_x}/{xw} \n"                       
                      f"s_y/y: {s_y}/{y} \n e_y/y+h: {e_y}/{yh} \n"
                      f"w: {w}\n h: {h}\n x: {x}\n y: {y}\n  half_scale: {half_scale}\n start_x: {start_x}\n start_y: {start_y}\n end_x: {end_x}\n end_y: {end_y}\n"
                      f"padding height: {padding_height} \n"
                      f"padding width: {padding_width} \n"
                      )

#                if w_rot > threshold and h_rot > threshold:  # LIMIT THE OBJECT SIZE
                cv2.imwrite(
                    str(Path(output_folder) /
                            "particles" /
                            f"particle_{str(img_name)}_{counter_total}.png"),
                            particle
                            )
                cv2.imwrite(
                        str(Path(output_folder) /
                            "mask_particle" /
                            f"particle_mask{str(img_name)}_{counter_total}.png"),
                            particle_mask
                            )
                cv2.imwrite(
                        str(Path(output_folder) /
                            "contoured_particle" /
                            f"particle_{str(img_name)}_{counter_total}.png"),
                            particle_contour
                            )
                cv2.imwrite(
                        str(Path(output_folder) /
                            "nonscaled_particle" /
                            f"particle_{str(img_name)}_{counter_total}.png"),
                            particle_nonscaled
                            ) 
                ## calculate the aspect ratio of particles that is defined by the ratio of the 2 length of the bounding box that encapsulates the particle.
                aspect_ratio = max(h_rot, w_rot) / min(h_rot, w_rot)
                        
                    ## calculate the size of the particle
                    # AED - Area Equivalent Diameter converted to um
                aed = 2 * math.sqrt(particle_area3 / math.pi)
                    
		    # Min diameter is the diameter of the circle calculated from the area
                min_dia = 2 * math.sqrt((w_rot * h_rot) / math.pi) * px_res

                    # maximum_dimension / ferret / DIAMETER of the circle 
                _, radius = cv2.minEnclosingCircle(contour_box)
                max_dim =  radius * px_res * 2

                    # Area ratio
                area_ratio = particle_area3 / (math.pi/4 * max_dim**2)
                    
                    # Number of particles on the image
                sum_particles = counter

#                    print("particle EXPORTED \n")       

#                else:
#                    file.write(f"{str(img_name)}_{counter} "+ "particle smaller than 10um"+ "\n")
#                    print(f"\n {str(img_name)}_{counter} NOT exported \n")

				# Write the particle numbers on the image                
                cv2.putText(
                    contour_image,
                    f"{counter_total}",
                    (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2,
                    cv2.LINE_AA
                    )                

                cv2.imwrite(str(Path(output_folder) /
                    "bbox" /
                    f"box_{image_files}"),
                    contour_image
                    )

                h, w = img_u8c.shape[:2]    
                sampling_area = h * w * px_res * px_res

                file.write(f"{str(img_name)}_{counter_total} " +
                            f"{aspect_ratio} " +
                            f"{aed} " +
                            f"{min_dia} " +
                            f"{max_dim} " +
                            f"{x} " +
                            f"{y} "+
                            f"{border} " +
                            f"{area_ratio} " +
                            f"{particle_area} " +
                            f"{particle_area2} " +
                            f"{particle_area3} " +
                            f"{particle_area_0} " +
                            f"{closed[0]} " +
                            f"{sampling_area}" +
                            "\n" 
                            )

                counter += 1
                counter_total += 1

			### FIND THE SMALLEST CIRCLE/FERRET DIAMETER
            for bounding_circle in contours:
               
                (circle_x, circle_y), radius = cv2.minEnclosingCircle(bounding_circle)
                center = (int(circle_x), int(circle_y))
                radius = int(radius)
                    
                contour_circle = cv2.circle(contour2, center, radius, (0, 0, 255), 1)
            
                cv2.imwrite(str(Path(output_folder) /
                    "circle" /
                    f"circle_{image_files}"),
                    contour_circle
                    )
                
            img_3ch = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            overlay = _merge_images(img_3ch[:, :, 2], mask)
            img_3ch[:, :, 2] = np.where(overlay, 255, img_3ch[:, :, 2])
            cv2.imwrite(
                str(Path(output_folder) /
                    "overlay" /
                    f"overlay_{image_files}"),
                    img_3ch
                    )
