from tensorflow import keras
import tensorflow as tf
import os


class OverlayWriter(keras.callbacks.Callback):
    def __init__(self, tag, logdir, samples):
        # self.__super__.init()
        self.tag = tag
        self.logdir = os.path.join(logdir, "images")
        self.samples = samples

        self._create_writer()

    def _create_writer(self):
        self.writer = tf.summary.create_file_writer(
            self.logdir,
            max_queue=None,
            flush_millis=None,
            filename_suffix=None,
            name=None,
            experimental_trackable=False,
        )

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.samples[0])
        samples_float = self.samples[0]

        sample_3ch = tf.image.grayscale_to_rgb(samples_float, name=None)

        red_ch = tf.where(
            predictions[:, :, :, 0] > 0.5,
            tf.ones_like(predictions[:, :, :, 0]),
            sample_3ch[:, :, :, 0],
        )

        samples_overlay = tf.stack(
            [red_ch, sample_3ch[:, :, :, 1], sample_3ch[:, :, :, 2]], axis=3
        )

        with self.writer.as_default():
            tf.summary.image(self.tag, samples_overlay, step=epoch)

        red_ch_annot = tf.where(
            self.samples[1][:, :, :, 0] > 0,
            tf.cast(self.samples[1][:, :, :, 0], tf.float32),
            sample_3ch[:, :, :, 0],
        )

        samples_overlay_gt = tf.stack(
            [
                red_ch_annot,
                sample_3ch[:, :, :, 1],
                sample_3ch[:, :, :, 2],
            ],
            axis=3,
        )

        with self.writer.as_default():
            tf.summary.image(self.tag + "_gt", samples_overlay_gt, step=epoch)
