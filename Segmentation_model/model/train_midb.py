# https://keras.io/examples/vision/oxford_pets_image_segmentation/


import os
import dataset
from tensorflow import keras
from args import parse_args
import sys
import os
from writer import OverlayWriter
import binary_losses
import tensorflow as tf
import metrics
import segmentation_models as sm
import onnx
import tf2onnx
#import logger
#from tensorflow.keras import metrics
#from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, Recall, Precision
from tensorflow.keras.callbacks import EarlyStopping

model_size = (
    960,
    1280,
)
num_classes = 1
batch_size = 4
epochs = 20 


early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min", # Stop when the monitored quantity stops decreasing
            verbose=1, # Display log messages when early stopping is triggered
            restore_best_weights=True # REstore the model weights from the epoch with the best value of the monitored quality
            )

def main(args):
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    # model = get_model(img_size, num_classes)
    model = sm.Linknet(
        "efficientnetb7",
        input_shape=(model_size[0], model_size[1], 1),
        classes=num_classes,
        activation="sigmoid",
        encoder_weights=None,
        decoder_filters=(256, 128, 64, 32, 16),
    )
    model.summary()

    train_paths, eval_paths = dataset.get_data_paths(args.data_folder)

    # Instantiate data Sequences for each split
    train_gen = dataset.CropsDataset(
        batch_size,
        model_size,
        train_paths,
        do_augment=True,
    )
    val_gen = dataset.CropsDataset(
        batch_size,
        model_size,
        eval_paths,
    )
    print("train set length: ", len(train_gen))
    print("Validation set length: ", len(val_gen))

    model.compile(
        optimizer="adam",
        loss=binary_losses.binary_tversky_loss(beta=0.05),  # increase recall
        metrics=[metrics.accuracy, metrics.fp, metrics.recall, metrics.prec],
    )
    input_signature = (
        tf.TensorSpec(
            (None, model_size[0], model_size[1], 1),
            tf.float32,
            name="image_input",
        ),
    )

    os.mkdir(args.output_folder)
#    state_logger = logger.StateLogger(args.output_folder)
#    state_logger.dump(args)

    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
    onnx.save(onnx_model, os.path.join(args.output_folder, "model_init.onnx"))

    def scheduler(epoch, lr):
        if epoch < epochs - 4:
            return lr
        else:
            return lr * tf.math.exp(-0.1)
    
    

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_folder, "crystal_model.h5"), save_best_only=True
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_folder, "logs"), histogram_freq=1
        ),
        OverlayWriter(
            "Validation images", os.path.join(args.output_folder, "logs"), val_gen[0]
        ),
        OverlayWriter(
            "Train images", os.path.join(args.output_folder, "logs"), train_gen[0]
        ),
        keras.callbacks.LearningRateScheduler(scheduler),
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=2, # After how many epoch should it stop
            mode="min", # Stop when the monitored quantity stops decreasing
            verbose=1, # Display log messages when early stopping is triggered
            restore_best_weights=True # REstore the model weights from the epoch with the best value of the monitored quality
            )
    ]

    # Train the model, doing validation at the end of each epoch.
    model.fit(train_gen, 
        epochs=epochs, 
        validation_data=val_gen, 
        callbacks=callbacks
        )

    model.save(os.path.join(args.output_folder, "model_file"))
    input_signature = (
        tf.TensorSpec(
            (None, model_size[0], model_size[1], 1),
            tf.float32,
            name="image_input",
        ),
    )
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
    onnx.save(onnx_model, os.path.join(args.output_folder, "model.onnx"))


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
