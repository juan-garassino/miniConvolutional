import matplotlib.pyplot as plt
import numpy as np

def plot(train_generator, idx_labels):

    image_batch, label_batch = next(iter(train_generator))

    fig, axes = plt.subplots(8, 4, figsize=(10, 20))

    axes = axes.flatten()

    for img, lbl, ax in zip(image_batch, label_batch, axes):
        ax.imshow(img)
        label_ = np.argmax(lbl)
        label = idx_labels[label_]
        ax.set_title(label)
        ax.axis('off')
    plt.show()
