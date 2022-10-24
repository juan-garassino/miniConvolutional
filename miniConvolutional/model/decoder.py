from tensorflow.keras.layers import Reshape, Conv2DTranspose, Dense
from tensorflow.keras import Sequential

def build_decoder(latent_dimension):
    # $CHALLENGIFY_BEGIN
    decoder = Sequential()

    decoder.add(Dense(7 * 7 * 8, activation='tanh', input_shape=(latent_dimension, )))
    decoder.add(Reshape((7, 7, 8)))  # no batch axis here
    decoder.add(
        Conv2DTranspose(8, (2, 2),
                        strides=2,
                        padding='same',
                        activation='relu'))

    decoder.add(
        Conv2DTranspose(1, (2, 2),
                        strides=2,
                        padding='same',
                        activation='relu'))
    return decoder
    # $CHALLENGIFY_END
