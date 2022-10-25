from tensorflow.keras import layers, Model


class Convolutional(Model):
    def __init__(self, input_dim):
        super(Convolutional, self).__init__()
        self.conv2d_initial = layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="glorot_uniform",
            padding="same",
            input_shape=(input_dim, input_dim, 3),
        )
        self.cov2d_mid = layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="glorot_uniform",
            padding="same",
        )
        self.cov2d_end = layers.Conv2D(
            128,
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="glorot_uniform",
            padding="same",
        )
        self.maxpool2d = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(
            256, activation="relu", kernel_initializer="glorot_uniform"
        )
        self.fc = layers.Dense(10, activation="linear", name="custom_class")

    def call(self, input_dim):
        x = self.conv2d_initial(input_dim)
        x = self.maxpool2d(x)
        x = self.cov2d_mid(x)
        x = self.maxpool2d(x)
        x = self.cov2d_end(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.fc(x)
        x = self.flatten(x)

        return x
