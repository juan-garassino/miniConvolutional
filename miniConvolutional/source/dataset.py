import os
from colorama import Fore, Style
from tensorflow.data import Dataset
from tensorflow import convert_to_tensor, float32, int32
import random


def dataset_instance(train_images, train_labels, test_images, test_labels):

    images_shape = train_images.shape

    train_images, train_labels, test_images, test_labels = (
        convert_to_tensor(train_images, dtype=float32),
        convert_to_tensor(train_labels, dtype=int32),
        convert_to_tensor(test_images, dtype=float32),
        convert_to_tensor(test_labels, dtype=int32),
    )

    print("\nℹ️ " + Fore.CYAN + f"Images shape is {images_shape}" + Style.RESET_ALL)

    validation_dataset = Dataset.from_tensor_slices(
        (test_images[:500], test_labels[:500])
    )

    test_dataset = Dataset.from_tensor_slices((test_images[500:], test_labels[500:]))

    # Create an instance of dataset from raw numpy images and labels
    train_dataset = Dataset.from_tensor_slices((train_images, train_labels))

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

    # Datasets
    train_dataset = train_dataset.shuffle(50000).batch(
        int(os.environ.get("TRAIN_BATCH_SIZE"))
    )

    validation_dataset = validation_dataset.batch(500)

    test_dataset = test_dataset.batch(500)

    STEPS_PER_EPOCH = train_dataset_size / int(os.environ.get("TRAIN_BATCH_SIZE"))

    random.seed(2)

    random_sel = random.sample(train_idx, 25)

    return (train_dataset, validation_dataset, images_shape, random_sel, STEPS_PER_EPOCH)
