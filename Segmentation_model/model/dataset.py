import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import os
import random
from dataclasses import dataclass
from PIL import Image

FOLDERS = ["batch_1"]


@dataclass
class DataPaths:
    input_img_paths: list
    target_img_paths: list


def get_data_paths(base_dir, split_ratio=0.15):
    total_input_img_paths = []
    total_target_img_paths = []

    for f in FOLDERS:
        dataset_dir = os.path.join(base_dir, f)

        input_dir = os.path.join(dataset_dir, "cropped_original_png")
        target_dir = os.path.join(dataset_dir, "cropped_contours")

        input_img_paths = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".png")
            ]
        )
        target_img_paths = sorted(
            [
                os.path.join(target_dir, fname)
                for fname in os.listdir(target_dir)
                if fname.endswith(".png") and not fname.startswith(".")
            ]
        )

        # There are fewer annotations then training data or some of them are tiff, which is not handled yet
        # so for now these input images are ommited
        input_img_path_to_keep = []
        for input_fname in input_img_paths:
            fname_stripped = input_fname.split(".")[0].split("/")[-1]
            if any(
                fname_stripped in target_img_fname
                for target_img_fname in target_img_paths
            ):
                input_img_path_to_keep.append(input_fname)

        print(
            f"Warning: {len(input_img_paths) - len(input_img_path_to_keep)} input images are ommited"
        )

        img_non_zero = []
        target_non_zero = []
        for target_path, img_path in zip(target_img_paths, input_img_path_to_keep):
            with Image.open(target_path) as im:
                np_img = np.array(im)
                if (np_img == 0).any():
                    img_non_zero.append(img_path)
                    target_non_zero.append(target_path)

        total_input_img_paths += img_non_zero
        total_target_img_paths += target_non_zero

    assert len(total_input_img_paths) == len(
        total_target_img_paths
    ), f"{len(total_input_img_paths)} != {len(total_target_img_paths)}"

    random.Random(1337).shuffle(total_input_img_paths)
    random.Random(1337).shuffle(total_target_img_paths)

    print("Number of samples:", len(total_input_img_paths))

    split_index = int(len(total_input_img_paths) * split_ratio)

    train_paths = DataPaths(
        input_img_paths=total_input_img_paths[:-split_index],
        target_img_paths=total_target_img_paths[:-split_index],
    )

    eval_paths = DataPaths(
        input_img_paths=total_input_img_paths[-split_index:],
        target_img_paths=total_target_img_paths[-split_index:],
    )

    return train_paths, eval_paths


class CropsDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(
        self,
        batch_size,
        img_size,
        paths,
        do_augment=False,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.paths = paths
        self.do_augment = do_augment

    def __len__(self):
        return len(self.paths.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.paths.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.paths.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, input_path in enumerate(batch_input_img_paths):
            input_img = load_img(
                input_path, target_size=self.img_size, color_mode="grayscale"
            )
            input_img_np_norm_gray = np.array(input_img).astype(np.float32)
            h, w = input_img_np_norm_gray.shape

            norm_value = (
                np.iinfo(np.uint16).max
                if np.max(input_img_np_norm_gray) > np.iinfo(np.uint8).max
                else np.iinfo(np.uint8).max
            )
            x[j, 0:h, 0:w] = (
                np.expand_dims(input_img_np_norm_gray[:, :], axis=2) / norm_value
            )  # normalize data

        y = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            np_img = np.array(img, dtype=np.uint8) / 255
            h, w = np_img.shape
            y[j, 0:h, 0:w] = 1 - np_img

        return self.do_augmentation(x, y)

    @tf.function
    def do_augmentation(self, input_image, input_mask):
        input_mask = tf.expand_dims(input_mask, axis=3)

        if self.do_augment:
            CROP_H = 864
            CROP_W = 1152

            if tf.random.uniform(()) > 0.5:
                offset_height = int((self.img_size[0] - CROP_H) * tf.random.uniform(()))
                offset_width = int((self.img_size[1] - CROP_W) * tf.random.uniform(()))

                # use original image to preserve high resolution
                input_image = tf.image.crop_to_bounding_box(
                    input_image, offset_height, offset_width, CROP_H, CROP_W
                )

                input_mask = tf.image.crop_to_bounding_box(
                    input_mask, offset_height, offset_width, CROP_H, CROP_W
                )
                # resize
                input_image = tf.image.resize(input_image, self.img_size)
                input_mask = tf.image.resize(input_mask, self.img_size)
                input_mask = tf.cast(input_mask, tf.uint8)
            # random brightness adjustment illumination
            input_image = tf.image.random_brightness(input_image, 0.5, seed=1121)
            # random contrast adjustment
            input_image = tf.image.random_contrast(input_image, 0.5, 1.5)
            if tf.random.uniform(()) > 0.5:
                input_image = tf.image.flip_left_right(input_image)
                input_mask = tf.image.flip_left_right(input_mask)
            if tf.random.uniform(()) > 0.5:
                input_image = tf.image.flip_up_down(input_image)
                input_mask = tf.image.flip_up_down(input_mask)

            # rotation in 30° steps
            rot_factor = tf.cast(
                tf.random.uniform(shape=[], maxval=12, dtype=tf.int32), tf.float32
            )
            angle = np.pi / 60 * rot_factor
            input_image = tfa.image.rotate(input_image, angle)
            input_mask = tfa.image.rotate(input_mask, angle)

        return input_image, input_mask
