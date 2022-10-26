from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import Dataset

import os
import numpy as np
import random
from colorama import Fore, Style


def source_images(generator=False, mnist=False, cifar=False):

    if mnist == True:

        (train_images, train_labels), (
            test_images,
            test_labels,
        ) = fashion_mnist.load_data()

        print(
            f"The shape of the training data is: {train_images.shape} {train_labels.shape}"
        )

        print(
            f"The shape of the training data is: {test_images.shape} {test_labels.shape}"
        )

        return (train_images, train_labels), (test_images, test_labels)

    if generator == True:

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

    if cifar == True:
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        # Normalize pixel values to be between 0 and 1
        train_images, test_images = train_images / 255.0, test_images / 255.0

        # Plain text name in alphabetical order. https://www.cs.toronto.edu/~kriz/cifar.html

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
            + f"ℹ️ \n{counts}"
            + Style.RESET_ALL
        )

        validation_dataset = Dataset.from_tensor_slices(
            (test_images[:500], test_labels[:500])
        )

        test_dataset = Dataset.from_tensor_slices(
            (test_images[500:], test_labels[500:])
        )

        # Create an instance of dataset from raw numpy images and labels.
        train_dataset = Dataset.from_tensor_slices((train_images, train_labels))

        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#transformations_2

        train_dataset_size = len(list(train_dataset.as_numpy_iterator()))

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"Training data sample size: "
            + str(train_dataset_size)
            + Style.RESET_ALL
        )

        print(
            "\nℹ️ "
            + Fore.CYAN
            + f"Length of training labels: "
            + str(len(train_labels))
            + Style.RESET_ALL
        )

        train_idx = list(range(len(train_labels)))

        random.seed(2)

        random_sel = random.sample(train_idx, 25)

        train_dataset = train_dataset.shuffle(50000).batch(
            int(os.environ.get("TRAIN_BATCH_SIZE"))
        )

        validation_dataset = validation_dataset.batch(500)

        test_dataset = test_dataset.batch(500)

        STEPS_PER_EPOCH = train_dataset_size / int(os.environ.get("TRAIN_BATCH_SIZE"))

        return train_dataset, validation_dataset, random_sel, STEPS_PER_EPOCH
