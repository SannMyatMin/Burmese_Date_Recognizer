import tensorflow as tf
from tensorflow.keras import models, layers

class DayMonth_Predictor:
    def __init__(self, num_classes=32, shape=(64, 64, 3)):
        self.model       = None
        self.shape       = shape
        self.num_classes = num_classes

    def build_cnn(self):
        input = layers.Input(shape=self.shape)

        x = layers.Conv2D(32, 3, padding="same", activation="relu")(input)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        output = layers.Dense(self.num_classes, activation="softmax")(x)
        cnn_model = models.Model(input, output)
        self.model = cnn_model
        return self.model

    def compile_model(self):
        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = "sparse_categorical_crossentropy",
            metrics = ["accuracy"]
        )
        print("Model is compiled successfully")

    def get_callbacks(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.3, verbose=1)
        ]
        return callbacks
    
    def train_model(self, train_ds, val_ds, epochs):
        history = self.model.fit(
            train_ds,
            validation_data = val_ds,
            epochs = epochs,
            callbacks = self.get_callbacks()
        )
        print("Model is trained successfully")
        return history
