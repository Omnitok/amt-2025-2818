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

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
#from tensorflow.keras.datasets import fashion_mnist
#from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Do i need this to use pretrained models, like resnet?
#import dataset
from args import parse_args
import segmentation_models as sm


epochs = 50 
augment = 0.252577777772275 
print("rotation: ", augment)

# IMPORT THE DATASET
import pathlib
data_dir = "/home/stejan/classification/particles_v2" 
data_dir = pathlib.Path(data_dir).with_suffix('')

image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)

# CREATE DATASET
batch_size = 4
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
 
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# VISUALIZE THE DATA
#plt.style.use("dark_background")
plt.figure(figsize=(10, 10))
#for images, labels in train_ds.take(3):
#  for i in range(5):
#    ax = plt.subplot(3, 2, i + 1)
#    plt.imshow(images[i].numpy().astype("uint8"))
#    plt.title(class_names[labels[i]])
#    plt.axis("off")
plotted_categories = []
train_ds_shuffled = train_ds.shuffle(buffer_size=len(train_ds))
for images, labels in train_ds_shuffled:
    # Iterate over each image and label
    for image, label in zip(images, labels):
        # Check if the category has already been plotted
        if label.numpy() not in plotted_categories:
            # Plot the image
            ax = plt.subplot(2, 3, len(plotted_categories) + 1)
            plt.imshow(image.numpy().astype("uint8"))
            plt.title(class_names[label], fontsize=25)
            plt.axis("off")
            plotted_categories.append(label.numpy())

        if len(plotted_categories) == 11:
            break
    if len(plotted_categories) == 11:
        break
plt.tight_layout()
plt.savefig("classes.png", transparent=False, dpi=300)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# CONFIGURE THE DATASET FOR PERFORMANCE
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# STANDARDIZE THE DATA
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

#CREATE THE MODEL
num_classes = len(class_names)

#model = Sequential([
#  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#  layers.Conv2D(16, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(32, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(64, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Flatten(),
#  layers.Dense(128, activation='relu'),
#  layers.Dense(num_classes)
#])

model_size = (
    180,
    180
    )

model = sm.Linknet(
    "resnet50",
    input_shape=(model_size[0], model_size[1], 1),
    classes=num_classes,
    activation="sigmoid",
    encoder_weights=None,
    decoder_filters=(256,128,64,32,16),
    )
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

#%% DATA AUGMENTATION
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(augment),
    layers.RandomZoom(augment),
  ]
)
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(5):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(2, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
plt.suptitle("Augmented images", fontsize=25)
plt.savefig("augmentation.png", dpi=300)

# DROPOUT
#model = Sequential([
#  data_augmentation,
#  layers.Rescaling(1./255),
#  layers.Conv2D(16, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(32, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Conv2D(64, 3, padding='same', activation='relu'),
#  layers.MaxPooling2D(),
#  layers.Dropout(0.2),
#  layers.Flatten(),
#  layers.Dense(128, activation='relu'),
#  layers.Dense(num_classes, name="outputs")
#])
#%% COMPILE AND TRAIN THE MODEL
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
#%%
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right', fontsize=14)
plt.title('Accuracy', fontsize=25)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right', fontsize=14)
plt.title('Loss', fontsize=25)
plt.tight_layout()
plt.savefig("training.png", transparent=False,  dpi=300)

## SAVE MODEL
model_path = "/home/stejan/classification/model/"
model.save("classifier_v2.h5")

#%% MAKE SOME PREDICTIONS
#crystal_path = "/home/stejan/snow_crystal_segmentation/particles/"
#crystal_path = "/home/stejan/image_analysis/001.png"
#import matplotlib.image as mpimg
#import os

# load the mdoel
#model = tf.keras.models.load_model("/home/stejan/classification/classifier.h5")

save_dir = "/home/stejan/classification/result_v2/"
predictions = []
cmatrix = []

#for root, dirs, files in os.walk(crystal_path):
#    for filename in files:
#        file_path = os.path.join(root, filename)
#        image = mpimg.imread(file_path)
#        img = tf.keras.utils.load_img(file_path, target_size=(img_height, img_width))
#        img_array = tf.keras.utils.img_to_array(img)
#        img_array = tf.expand_dims(img_array, 0)
#        pred = model.predict(img_array)
#        score = tf.nn.softmax(pred[0])
#        plt.imshow(image)
#        plt.title("{} with {:.2f} confidence".format(class_names[np.argmax(score)], 100 * np.max(score)))
#        save_path = save_dir + filename
 #       plt.savefig(save_path)
  #      print(f"{save_path} saved")
   #     plt.close()
    #    predictions.append(pred)
        # c_matrix = tf.math.confusion_matrix(
        #     labels,
        #     predictions,
        #     num_classes=None,
        #     weights=None,
        #     dtype=tf.dtypes.double,
        #     name=None
        # )

#%% PREDICT ON THE DATA
train_data = train_ds #/ 255.0
test_data = val_ds #/ 255.0

y_pred = model.predict(test_data)
ypred_classes = np.argmax(y_pred, axis=1)

#%% CONFUSION MATRIX
# compute the confusion matrix
#cm = confusion_matrix(test_labels, y_pred_classes, labels=np.arange(len(class_names)))

# normalize the confusion matrix
#cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

# Display the confusion matrix
#disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
#fig, ax = plt.subplots(figsize=(12, 12))
#disp.plot(cmap="Blues", ax=ax, xticks_rotation="vertical")
#plt.savefig("cmx.png")
#plt.show()

#class_name_list = []
#x_test = train_ds
#y_test = class_names

# create a dict to map classes to integer
#class_name_id = {class_name: i for i, class_name in enumerate(class_names)}
#labels_int = []

#for images, labels in source:
#    for label in labels.numpy():
#        class_name = class_names[label]
#        class_id = class_name[class_name]
#        labels_int.append(class_id)
#        class_name_list.append(class_name)

# make the predictions
#print("prediction: ")
#y_pred = model.predict(x_test)

# convert predicted probabilities to class labels
#y_pred_labels = np.argmax(y_pred, axis=1)
#predicted_labels = np.argmax(y_pred















