import tensorflow as tf
from tensorflow.keras import models, layers

class DayMonth_Predictor:
    def __init__(self, num_classes=32, shape=(64, 64, 3)):
        self.shape = shape
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
        return models.Model(input, output)

