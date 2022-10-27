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
            input_shape=input_dim,
        )
        self.cov2d_middle = layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="glorot_uniform",
            padding="same",
        )
        self.cov2d_final = layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="glorot_uniform",
            padding="same",
        )
        self.maxpool2d = layers.MaxPooling2D(pool_size=(2, 2))

        self.flatten = layers.Flatten()

        self.dense_initial = layers.Dense(
            256, activation="relu", kernel_initializer="glorot_uniform"
        )

        self.dense_middle = layers.Dense(
            128, activation="relu", kernel_initializer="glorot_uniform"
        )

        self.dense_out = layers.Dense(10, activation="linear", name="custom_class")

    def call(self, input_dim):

        x = self.conv2d_initial(input_dim)

        x = self.maxpool2d(x)

        x = self.cov2d_middle(x)

        x = self.maxpool2d(x)

        # x = self.cov2d_final(x)

        x = self.flatten(x)

        x = self.dense_initial(x)

        # x = self.dense_middle(x)

        x = self.dense_out(x)

        return x
