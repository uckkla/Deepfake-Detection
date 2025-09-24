import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class DeepfakeDetector:
    def __init__(self, input_shape=(256, 256, 3), batch_size=32):
        self.input_shape = input_shape
        self.batch_size=batch_size
        self.model = self.build_model()

    def build_model(self):
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape
        )
        # Freeze initially
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model


    def load_data(self, data_dir, test_size=0.2, val_size=0.2, random_state=20):
        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")

        real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
        fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]

        file_paths = np.array(real_files + fake_files)
        labels = np.array([0]*len(real_files) + [1]*len(fake_files))

        # Stratified split to maintain class balance
        # train_test_split done twice to allow for validation set
        X_temp, X_test, y_temp, y_test = train_test_split(
            file_paths, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )

        val_ratio = val_size / (1-test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # Used to prevent overfitting
    def augment_image(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_brightness(image, max_delta=0.1)
        #image = tf.image.random_zoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))
        return image

    def preprocess_image(self, path, augment=False):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.input_shape[:2])
        image = image / 255.0  # Normalisation

        if augment:
          image = self.augment_image(image)
        return image

    def data_generator(self, file_paths, labels, augment=False):
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.map(lambda x, y: (self.preprocess_image(x, augment=augment), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def train(self, train_data, val_data, epochs_max=100, model_dir=""):
        train_ds = self.data_generator(*train_data, augment=True)
        val_ds = self.data_generator(*val_data, augment=False)
        model_dir = os.path.join(model_dir,"best_model.keras")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                model_dir,
                monitor="val_loss",
                save_best_only=True
            )
        ]

        full_history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": []
        }

        # Phase 1: Train only the top layers (frozen base)
        print("\nPhase 1: Training top layers only")
        history1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=5,  # Initial phase epochs
            callbacks=callbacks
        )
        for key in full_history.keys():
            full_history[key] += history1.history.get(key, [])

        # Phase 2: Unfreeze some intermediate layers
        print("\nPhase 2: Unfreezing middle layers")
        self.model.layers[0].trainable = True
        for layer in self.model.layers[0].layers[:150]:  # Freeze first 150 layers
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        history2 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=5,
            epochs=10,  # Second phase epochs
            callbacks=callbacks
        )
        for key in full_history.keys():
            full_history[key] += history2.history.get(key, [])

        # Phase 3: Unfreeze all layers
        print("\nPhase 3: Fine-tuning all layers")
        for layer in self.model.layers[0].layers:
            layer.trainable = True
        
        # Even lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        history3 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=10,
            epochs=epochs_max,
            callbacks=callbacks
        )
        for key in full_history.keys():
            full_history[key] += history3.history.get(key, [])

        return full_history

def min_int_type(arg):
    ivalue = int(arg)
    if ivalue < 15:
        raise argparse.ArgumentTypeError(f"Minimum value is 15, got {ivalue}")
    return ivalue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Model")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the dataset folder containing 'real/' and 'fake/' subfolders"
    )
    parser.add_argument(
        "--model_dir", type=str, default = "",
        help="Path to save the best model (default: current directory)"
    )
    parser.add_argument(
        "--epochs", type=min_int_type, default=100,
        help="Total number of epochs (minimum: 15, default: 100)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size for training (default: 128)"
    )
    args = parser.parse_args()

    # Initialise deepfake detector model
    detector = DeepfakeDetector(input_shape=(224, 224, 3), batch_size=args.batch_size)

    # Load dataset and split into train/val/test
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = detector.load_data(args.data_dir)

    # Step 3: Train the model
    history = detector.train((train_X, train_y), (val_X, val_y), epochs_max=args.epochs, model_dir=args.model_dir)

    # Evaluate the trained model on the test set and print accuracy
    test_dataset = detector.data_generator(test_X, test_y)
    test_loss, test_acc = detector.model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
