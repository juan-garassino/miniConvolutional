from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def build_encoder(latent_dimension):
    """returns an encoder model, of output_shape equals to latent_dimension"""
    encoder = Sequential()

    encoder.add(Conv2D(8, (2, 2), input_shape=(28, 28, 1), activation="relu"))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(16, (2, 2), activation="relu"))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(32, (2, 2), activation="relu"))
    encoder.add(MaxPooling2D(2))

    encoder.add(Flatten())
    encoder.add(Dense(latent_dimension, activation="tanh"))

    return encoder
