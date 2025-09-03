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
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import image_dataset_from_directory

# DO I NEED THIS IN ORDER TO USE THE PRETRAINED CNN-S?
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

epochs = 50 
augment = 0.252577777772275 
patience = 10

trainable = False 
save_name = "classifier_250410nt4cat.keras" 

# IMPORT THE DATASET
import pathlib
data_dir = "/amt-2025-2818/Classification_model/particles" 
data_dir = pathlib.Path(data_dir).with_suffix('')

image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)

# CREATE DATASET
batch_size = 16 
img_height = 200 
img_width = 200

def resize_image(image, label):
    image = tf.image.resize_with_pad(image, img_height, img_width)
    return image, label

train_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# save class names
class_names = train_ds.class_names

train_ds = train_ds.map(resize_image)

val_ds = image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = val_ds.map(resize_image)

print(f"{class_names=}")

# VISUALIZE THE DATA
#plt.style.use("dark_background")
plt.figure(figsize=(10, 10))
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
plt.savefig("classes_" + save_name[11:17] + ".png", bbox_inches="tight", transparent=False, dpi=300)
plt.close()

for image_batch, labels_batch in train_ds:
  print(f"{image_batch.shape=}")
  print(f"{labels_batch.shape=}")
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

model_size = (
    200,
    200
    )
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, ResNet152, EfficientNetB7

# Load a pre-trained ResNet model and customize it to your input size
base_model = tf.keras.applications.EfficientNetB7(
    input_shape=(img_height, img_width, 3),
    include_top=False,  # Exclude the top classification layer
    weights='imagenet'  # Load weights pre-trained on ImageNet
)

# Freeze the layers of the pre-trained model to avoid training them
base_model.trainable = trainable

# Add new layers on top of the pre-trained model
model = tf.keras.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),

    base_model,  # Pretrained feature extractor

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),  # Optional: helps with downsampling

    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
tf.keras.utils.plot_model(model, show_shapes=True, to_file=("model_architecture_" + save_name[11:17]  + ".png"))

#% DATA AUGMENTATION
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal"),
#                      input_shape=(img_height,
#                                  img_width,
#                                  3)),
    layers.RandomRotation(augment),
    layers.RandomZoom(augment),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(factor=0.3)
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
plt.savefig("augmentation_" + save_name[11:17] + ".png", dpi=300)
plt.close()

# Define early stopping callback
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',   # Monitor the validation loss
    patience=patience,           # Number of epochs with no improvement to wait before stopping
    mode='min',           # Stop when the monitored quantity stops decreasing ('min' mode)
    verbose=1,            # Display log messages when early stopping is triggered
    restore_best_weights=True  # Restore the model weights from the epoch with the best value of the monitored quantity
)

#% COMPILE AND TRAIN THE MODEL
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])#, Precision(), Recall()])
model.summary()

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[early_stopping]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
#%%
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(range(len(acc)), acc, label='Training Accuracy')
plt.plot(range(len(acc)), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right', fontsize=14)
plt.title('Accuracy', fontsize=25)

plt.subplot(2, 1, 2)
plt.plot(range(len(loss)), loss, label='Training Loss')
plt.plot(range(len(loss)), val_loss, label='Validation Loss')
plt.legend(loc='upper right', fontsize=14)
plt.title('Loss', fontsize=25)
plt.tight_layout()

plt.savefig("training_" + save_name[11:17] + ".png", transparent=False,  dpi=300)

## SAVE MODEL
model_path = "/amt-2025-2818/Classification_model/"
model.save(save_name)

#%% MAKE SOME PREDICTIONS
save_dir = "/amt-2025-2818/Classification_model/results/"
predictions = []
cmatrix = []

#%% PREDICT ON THE DATA
train_data = train_ds #/ 255.0
test_data = val_ds #/ 255.0

y_pred = model.predict(test_data)
ypred_classes = np.argmax(y_pred, axis=1)
test_labels = np.concatenate([y.numpy() for _, y in val_ds])

#%% CONFUSION MATRIX
# compute the confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(test_labels, ypred_classes, labels=np.arange(len(class_names)))

# normalize the confusion matrix
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

# Display the CM with counts and ratio
fig, ax = plt.subplots(figsize=(12,12))
im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues")

plt.colorbar(im, ax=ax)

# add class labels
ax.set(xticks=np.arange(len(class_names)),
       yticks=np.arange(len(class_names)),
       xticklabels=class_names,
       yticklabels=class_names,
       title="Confusion matrix",
       ylabel="True label",
       xlabel="Predicted label")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

thresh = cm_normalized.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, f"{cm[i, j]} \n ({cm_normalized[i, j]:.2f}]",
               ha="center", va="center",
               color="white" if cm_normalized[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("cmx_" + save_name[11:17] + ".png")
