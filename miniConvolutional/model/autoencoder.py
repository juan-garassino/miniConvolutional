from tensorflow.keras import Model

from tensorflow.keras.layers import Input


def build_autoencoder(encoder, decoder):
    inp = Input((28, 28, 1))
    encoded = encoder(inp)
    decoded = decoder(encoded)
    autoencoder = Model(inp, decoded)
    return autoencoder


def compile_autoencoder(autoencoder):
    # $CHALLENGIFY_BEGIN
    autoencoder.compile(loss="mse", optimizer="adam")
    # $CHALLENGIFY_END
