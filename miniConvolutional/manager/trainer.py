from miniConvolutional.source.data import source_images
from miniConvolutional.model.encoder import build_encoder
from miniConvolutional.model.decoder import build_decoder
from miniConvolutional.model.autoencoder import build_autoencoder, compile_autoencoder

import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = source_images()

X_train = train_images.reshape((60000, 28, 28, 1)) / 255.

X_test = test_images.reshape((10000, 28, 28, 1)) / 255.

latent_dim = 10

encoder = build_encoder(latent_dim)

decoder = build_decoder(latent_dim)

autoencoder = build_autoencoder(encoder, decoder)

print(encoder.summary())

print(decoder.summary())

print(autoencoder.summary())

compile_autoencoder(autoencoder)

autoencoder.fit(X_train, X_train, epochs=20, batch_size=32)

prediction = autoencoder.predict(
    X_train, verbose=0, batch_size=100
)  # you can now display an image to see it is reconstructed well

for i in range(3):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
    ax1.imshow(prediction[i].reshape(28, 28), cmap='Greys')
    ax2.imshow(X_train[i].reshape(28, 28), cmap='Greys')
    plt.show()

import seaborn as sns

X_encoded = encoder.predict(X_train, verbose=1)

plt.scatter(x=X_encoded[:300, 0],
            y=X_encoded[:300, 1],
            c=train_labels[:300],
            cmap='coolwarm')
