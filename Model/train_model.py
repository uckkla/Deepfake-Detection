import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class DeepfakeDetector:
    def __init__(self, input_shape=(256, 256, 3), batch_size=32):
        self.input_shape = input_shape
        self.batch_size=32
        self.model = self.build_model()
    
    def build_model(self):
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=self.input_shape
        )
        base_model.trainable = True

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model
    

    def load_data(self, data_dir, test_size=0.15, val_size=0.15, random_state=20):
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

        """
        image = load_img(path, target_size=self.input_shape[:2])
        image = img_to_array(image) / 255.0
        return image
        """
    def preprocess_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.input_shape[:2])
        return image / 255.0  # Normalisation

    
    def data_generator(self, file_paths, labels, batch_size=32):
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.map(lambda x, y: (self.preprocess_image(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
    
    def train(self, train_data, val_data, epochs=20):
        train_ds = self.data_generator(*train_data)
        val_ds = self.data_generator(*val_data)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "best_model.h5",
                monitor="val_loss",
                save_best_only=True
            )
        ]
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )
        return history











# Paths
source_dir = 'D:\Final Year Project\Processed Datasets\dfdc'
output_base = 'D:\Final Year Project\Processed Datasets\dfdc\split_dataset'

# Output structure
splits = ['train', 'val', 'test']
classes = ['real', 'fake']
split_ratios = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test

# Create output folders
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)