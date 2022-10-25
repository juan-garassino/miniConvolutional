import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot(train_generator, idx_labels):

    image_batch, label_batch = next(iter(train_generator))

    fig, axes = plt.subplots(8, 4, figsize=(10, 20))

    axes = axes.flatten()

    for img, lbl, ax in zip(image_batch, label_batch, axes):
        ax.imshow(img)
        label_ = np.argmax(lbl)
        label = idx_labels[label_]
        ax.set_title(label)
        ax.axis("off")
    plt.show()


def plot_prediction(prediction, X_train):

    for i in range(3):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2))
        ax1.imshow(prediction[i].reshape(28, 28), cmap="Greys")
        ax2.imshow(X_train[i].reshape(28, 28), cmap="Greys")
        plt.show()


def plot_latent_dimention(encoder, X_train, train_labels):

    X_encoded = encoder.predict(X_train, verbose=1)

    plt.scatter(
        x=X_encoded[:300, 0],
        y=X_encoded[:300, 1],
        c=train_labels[:300],
        cmap="coolwarm",
    )
