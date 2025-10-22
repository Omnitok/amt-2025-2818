#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:34:11 2023

@author: stejan
ICE CRYSTAL CLASSIFICATION
FOLLOWING: https://www.tensorflow.org/tutorials/images/classification
"""
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pandas as pd
import matplotlib.image as mpimg
import os
import cv2
import shutil

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from datetime import datetime, timedelta, date
from pathlib import Path
from astropy.table import Table
import re

debug = 0 # 0 or 1

source_path = "/amt-2025-2818/Segmentation_model/step3_output/<INPUT DIR>"
save_dir = "/amt-2025-2818/Segmentation_model/<SAVE DIR>"

flight = Path(source_path).name

# IMPORT DATASET
import pathlib
data_dir = "/amt-2025-2818/Classification_model/particles"
data_dir = pathlib.Path(data_dir).with_suffix('')
batch_size = 16 
img_width = 200 
img_height = 200

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

#%% LOAD THE MODEL
model_name = "classifier_1229nt4cat.keras"
model = tf.keras.models.load_model(model_name)
# The "old" classifiers (_0509, _0510, _0,5) works with tensorflow and python 3.8 environment.
classification_version = model_name[len("classifier_") : -len(".keras")]

#class_names = train_ds.class_names
class_names = ['columns', 'compact', 'plates', 'rosettes']
print(class_names)

import glob
crystal_path = os.path.join(source_path, "particles")
particle_mask_path = os.path.join(source_path, "mask_particle")
particle_dataset_path = glob.glob(os.path.join(source_path, "particles_*.txt"))[0]

# extract the model version
version_path = Path(particle_dataset_path)
segmentation_version = version_path.stem.split("_")[2:]

#%% PREDICT ON NEW DATA
save_dir_path = Path(save_dir)

if save_dir_path.exists():
    shutil.rmtree(save_dir)
    print("\n Folder exists, deleting...")
os.mkdir(save_dir)
print(f"\n Creating folder: {save_dir} \n")

columns = ["Particle_ID", "category", "category_uncertain", "aed", "min_dia",
           "max_dim", "aspect_ratio", "area_ratio", "particle_area", "particle_area_0",
           "x", "y", "P", "height", "T", "RH", "border", *class_names, "original_image_size"]

# create the dataframe
data = pd.DataFrame(columns=columns)

# To ECSV
ecsv_table = Table(names = columns,
                   dtype = ["str"] * 3 + ["f8"] * 19)
ecsv_table.meta["author"] = "Stenszky Janos"
ecsv_table.meta["classification model"] = classification_version
ecsv_table.meta["segmentation model"] = segmentation_version
ecsv_table.meta["created on"] = date.today().isoformat() 
ecsv_table[columns[0]].info.description = "Containing the time of the recording and the which particle on the image"
ecsv_table[columns[1]].info.description = "Highest probability of the 4 category"
ecsv_table[columns[2]].info.description = "If no probability reaches 30%, category=uncertain"
ecsv_table[columns[3]].info.description = "Area equivalent diameter in um"
ecsv_table[columns[4]].info.description = "Diagonal length of the bounding box in um"
ecsv_table[columns[5]].info.description = "diameter of smallest circle that encapsulates the particle in um"
ecsv_table[columns[6]].info.description = "Ratio of bounding box(width/length)"
ecsv_table[columns[7]].info.description = "Area ratio of the particle, Area of the object compared to a perfect circle with the same diameter "
ecsv_table[columns[8]].info.description = "particle area in um"
ecsv_table[columns[9]].info.description = "area where pixel values = 0, in um"
ecsv_table[columns[10]].info.description = "top-left corner x-coordinate of the bounding box on the image"
ecsv_table[columns[11]].info.description = "top-left corner y-coordinate of the bounding box on the image"
ecsv_table[columns[12]].info.description = "pressure, hpa"
ecsv_table[columns[13]].info.description = "altitude, m"
ecsv_table[columns[14]].info.description = "temperature, C"
ecsv_table[columns[15]].info.description = "relative humidity %"
ecsv_table[columns[16]].info.description = "Is particle on the border?"
ecsv_table[columns[17]].info.description = f"probability of {class_names[0]}"
ecsv_table[columns[18]].info.description = f"probability of {class_names[1]}"
ecsv_table[columns[19]].info.description = f"probability of {class_names[2]}"
ecsv_table[columns[20]].info.description = f"probability of {class_names[3]}"
ecsv_table[columns[21]].info.description = f"Sampling area in um2"

# Import particle data from the Segmentation model
particle_dataset = pd.read_csv(particle_dataset_path, delimiter=' ')
particle_dataset["time"] = pd.to_datetime(particle_dataset["Particle_ID"].str[:15], format="%Y%m%d_%H%M%S")

# Import PTU and ETAG data
ptu_dataset_path = "/home/stejan/Desktop/ballon/data/ptu_tot.txt"
ptu_dataset = pd.read_csv(ptu_dataset_path, delimiter="\t", names=['time_og', 'P', 'height', 'T', 'RH'])
# convert the date format
ptu_dataset["time_dt"] = pd.to_datetime(ptu_dataset["time_og"], format="%Y-%m-%d %H:%M:%S")
ptu_dataset["time"] = ptu_dataset["time_dt"].dt.strftime('%Y%m%d_%H%M%S')

#plt.style.use("dark_background")
 # Counters
n = 0
skipped_no_particles = 0
skipped_small_images = 0
skipped_bad_images = 0

# WALK THORUGH THE FOLDER FILE-BY FILE
for root, dirs, files in os.walk(crystal_path):
    # Sort the files to alphabetic order
    sorted_files = sorted(files)
    print("Sorted_files_length:", len(sorted_files))
    
    for filename in sorted_files:
        file_path = os.path.join(root, filename)
        
        ## CREATE THE IMAGE
        try:
            image = mpimg.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Skipping {filename} due to image read error: {e}")
            skipped_bad_images += 1
            continue

        img = tf.keras.utils.load_img(file_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Run the model on it
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            print(f"Skipping {filename} because image is too small")
            skipped_small_images += 1
            continue

        # PREDICTING ON THE DATA
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        plt.imshow(image)
          
          # Write the top 3 results
        top_indices = np.argsort(score)[::-1][:]
        all_classes = class_names

            # FIND A WAY TO ENUMERATE THE CLASSES, AND ASSIGN THE PROBABILITY FOR EACH CLASS THAN EXPORT IT TO THE DATASET
        all_confidences = score
        title_lines = (["{} ({:.2f}%)".format(c, 100 * s) for c, s in zip(all_classes, all_confidences)])
        title = "\n".join(title_lines)
        plt.title(title)
        save_path = os.path.join(save_dir, filename)
        save_path2 = save_dir + "circle_" + filename
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

          # Find the calculated diameters in particle_dataset
        filtered_particles = particle_dataset[~particle_dataset['Particle_ID'].str.endswith('_000')] # - GET RID OF PARTICLES THAT SMALLER THAN CRITERIA
        # Find the particle ID-s no matter how many digits does it has.
        match = re.match(r"(\d{8}_\d{6}_\d{3}_\d+)", filename[9:])
        if match:
            particle_id_base = match.group(1)
            matching_particle_row = filtered_particles[filtered_particles["Particle_ID"].str.startswith(particle_id_base)]
        else:
            matching_particle_row = None
        
        if debug:
            print(f"{particle_dataset['Particle_ID'].str[:21]=}")
            print(f"{filename=}")
            print(f"{filename[9:30]=}")
            print(f"{particle_id_base=}")

        # extract the row if found
        pid = filename[9:-4]
        if matching_particle_row.empty:
            print(f"Skipping {filename}: No matching particle ID: {pid}")
            skipped_no_particles += 1
            continue
     
        extracted_particle_row = matching_particle_row.values[0]
        aspect_ratio = extracted_particle_row[1]
        aed = extracted_particle_row[2]
        area_ratio = extracted_particle_row[8]
        min_dia = extracted_particle_row[3]
        max_dim = extracted_particle_row[4]
        x = extracted_particle_row[5]
        y = extracted_particle_row[6]
        border = extracted_particle_row[7]
        particle_area = extracted_particle_row[9]
        particle_area_0 = extracted_particle_row[12]
        sampling_area = extracted_particle_row[14]

        mask = particle_mask_path + "particle_mask" + filename[9:]
#            ferret = calculate_ferret(mask)
            
          # find the corresponding PTU data
        # TIME CORRECTION : 47mm according to Thomas's calculation
        time_correction = 47
        time_str = filename[9:24]
            # Define the time format
        time_format = "%Y%m%d_%H%M%S"
            # Subtract seconds for time correction
        new_time = datetime.strptime(time_str, time_format) - timedelta(seconds=time_correction)
        # Convert back to string
        new_time_str = new_time.strftime(time_format)
            
        if debug:
            print(filename)
            print(f"new_time: {new_time}")
            print(f"new_time_str: {new_time_str}")
            print(f"{n=}")
            print(f"{max_dim=}")
            print(f"{particle_area=}")
            print(f"{pid=}")

            # Find matching rows in the dataset
        matching_ptu_row = ptu_dataset[ptu_dataset['time'] == new_time_str]
        if not matching_ptu_row.empty:
            extracted_ptu_row = matching_ptu_row.values[0]
            p = extracted_ptu_row[1]
            height = extracted_ptu_row[2]
            t = extracted_ptu_row[3]
            rh = extracted_ptu_row[4]      
        else:
            p = 0
            height = 0
            t = 0
            rh = 0
        
        # write "uncertain of any of the values are below 30%
        category = class_names[np.argmax(score)] if np.max(score) > 0.3 else "uncertain"
          
            # write the data to the dataframe
        data.loc[len(data)] = [
            filename[9:-4],
            class_names[np.argmax(score)],
            category,
            aed,
            min_dia,
            max_dim,
            aspect_ratio,
            area_ratio,
            particle_area,
            particle_area_0,
            x,
            y,
            p,
            height,
            t,
            rh,
            border,
            score[0].numpy(),
            score[1].numpy(),
            score[2].numpy(),
            score[3].numpy(),
            sampling_area
            ]

        ecsv_table.add_row([
                filename[9:-4],
                str(class_names[np.argmax(score)]),
                str(category),
                aed,
                min_dia,
                max_dim,
                aspect_ratio,
                area_ratio,
                particle_area,
                particle_area_0,
                x,
                y,
                p,
                height,
                t,
                rh,
                border,
                score[0].numpy(),
                score[1].numpy(),
                score[2].numpy(),
                score[3].numpy(),
                sampling_area
                ])
        n += 1

print("--- Summary ---")
print(f"Processed: {n}")
print(f"Skipped (bad image): {skipped_bad_images}")
print(f"Skipped (small image): {skipped_small_images}")
print(f"Skipped (no particle match): {skipped_no_particles}")

## save the data
output_file = os.path.join(save_dir, "data.txt")
data.to_csv(output_file, sep="\t", index=False)

# save it to ecsv file
ecsv_output = os.path.join(save_dir, "classified_" + flight + ".csv")
ecsv_table.write(ecsv_output, format="ascii.ecsv", overwrite=True)
