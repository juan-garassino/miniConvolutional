from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from miniConvolutional.source.dataset import dataset_instance

import os
import numpy as np
from colorama import Fore, Style


def source_images(data=None):

    if data == "mnist":

        (train_images, train_labels), (
            test_images,
            test_labels,
        ) = fashion_mnist.load_data()

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"The shape of the training data is: {train_images.shape} {train_labels.shape}"
            + Style.RESET_ALL
        )

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"The shape of the testing data is: {test_images.shape} {test_labels.shape}"
            + Style.RESET_ALL
        )

        unique, counts = np.unique(train_labels, return_counts=True)

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"Training set with unique classes:"
            + "\n"
            + f"\nℹ️ {unique}"
            + Style.RESET_ALL
        )

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"Training set with distribution of class:"
            + "\n"
            + f"\nℹ️ {counts}"
            + Style.RESET_ALL
        )

        unique, counts = np.unique(test_labels, return_counts=True)

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"Testing set with unique classes:"
            + "\n"
            + f"\nℹ️ {unique}"
            + Style.RESET_ALL
        )

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"Testing set with distribution of class:"
            + "\n"
            + f"\nℹ️ {counts}"
            + Style.RESET_ALL
        )

        (train_images, train_labels), (test_images, test_labels,) = (
            train_images.reshape(
                (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
            ),
            train_labels,
        ), (
            test_images.reshape(
                (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
            ),
            test_labels,
        )

    if data == "cifar":

        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        # Normalize pixel values to be between 0 and 1
        train_images, test_images = train_images / 255.0, test_images / 255.0

        # Plain text name in alphabetical order. https://www.cs.toronto.edu/~kriz/cifar.html

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"The shape of the training data is: {train_images.shape} {train_labels.shape}"
            + Style.RESET_ALL
        )

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"The shape of the testing data is: {test_images.shape} {test_labels.shape}"
            + Style.RESET_ALL
        )

        CLASS_NAMES = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        unique, counts = np.unique(train_labels, return_counts=True)

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"Training set with unique classes:"
            + "\n"
            + f"\nℹ️ {unique}"
            + Style.RESET_ALL
        )

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"Training set with distribution of class:"
            + "\n"
            + f"\nℹ️ {counts}"
            + Style.RESET_ALL
        )

        unique, counts = np.unique(test_labels, return_counts=True)

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"Testing set with unique classes:"
            + "\n"
            + f"\nℹ️ {unique}"
            + Style.RESET_ALL
        )

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"Testing set with distribution of class:"
            + "\n"
            + f"\nℹ️ {counts}"
            + Style.RESET_ALL
        )

    if data == "generator":

        data_dir = get_file(
            "flower_photos",
            "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
            untar=True,
            extract=True,
            cache_subdir=os.path.join(
                os.environ.get("HOME"),
                "code",
                "juan-garassino",
                "miniSeries",
                "miniConvolutional",
                "data",
            ),
            cache_dir=os.path.join(
                os.environ.get("HOME"),
                "code",
                "juan-garassino",
                "miniSeries",
                "miniConvolutional",
                "data",
            ),
            archive_format="zip",
        )

        kwargs_datagen = dict(rescale=1.0 / 255, validation_split=0.20)

        kwargs_dataflow = dict(
            target_size=(int(os.environ.get("PIXEL")), int(os.environ.get("PIXEL"))),
            batch_size=int(os.environ.get("BATCH_SIZE")),
            interpolation="bilinear",
        )

        train_datagen = ImageDataGenerator(**kwargs_datagen)

        train_generator = train_datagen.flow_from_directory(
            data_dir,
            subset="training",
            shuffle=True,
            **kwargs_dataflow,
            class_mode="sparse",
        )

        valid_generator = train_datagen.flow_from_directory(
            data_dir, subset="validation", shuffle=False, **kwargs_dataflow
        )

        labels_idx = train_generator.class_indices

        idx_labels = dict((v, k) for k, v in labels_idx.items())

        for image_batch, labels_batch in train_generator:
            print(image_batch.shape)
            print(labels_batch.shape)
            # print(labels_batch)
            break

        for image_batch, labels_batch in valid_generator:
            print(image_batch.shape)
            print(labels_batch.shape)
            # print(labels_batch[0])
            break

        print(type(train_generator))
        print(type(valid_generator))
        print(idx_labels)
        print(labels_idx)

        return train_generator, idx_labels, valid_generator

    else:
        print('No data Loaded')
    (
        train_dataset,
        validation_dataset,
        dataset_shape,
        random_sel,
        STEPS_PER_EPOCH,
    ) = dataset_instance(train_images, train_labels, test_images, test_labels)

    return train_dataset, validation_dataset, dataset_shape, random_sel, STEPS_PER_EPOCH
