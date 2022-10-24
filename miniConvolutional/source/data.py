from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

import numpy as np
import matplotlib.pyplot as plt

def source_images(generator=False):

    if generator == False:

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        print(f'The shape of the training data is: {train_images.shape} {train_labels.shape}')

        print(f'The shape of the training data is: {test_images.shape} {test_labels.shape}')

        return (train_images, train_labels), (test_images, test_labels)

    else:

        data_dir = get_file('flower_photos',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
            untar=True, extract=True,
            cache_subdir=os.path.join(os.environ.get('HOME'), 'code', 'juan-garassino',
                                    'miniSeries', 'miniConvolutional', 'data'),
            cache_dir=os.path.join(os.environ.get('HOME'), 'code', 'juan-garassino', 'miniSeries',
            'miniConvolutional', 'data'), archive_format='zip')

        datagen_kwargs = dict(rescale=1./255, validation_split=0.20)

        train_datagen = ImageDataGenerator(**datagen_kwargs)

        IMAGE_SIZE = (224, 224) # Each image contains 224 by 224 by 3 pixels

        BATCH_SIZE = 32 # Each batch contains this sample count

        dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, interpolation="bilinear")

        train_generator = train_datagen.flow_from_directory(data_dir, subset="training", shuffle=True, **dataflow_kwargs)

        labels_idx = (train_generator.class_indices)

        idx_labels = dict((v,k) for k,v in labels_idx.items())

        for image_batch, labels_batch in train_generator:
            print(image_batch.shape)
            print(labels_batch.shape)
            break

        valid_datagen = train_datagen

        valid_generator = valid_datagen.flow_from_directory(data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

        return train_generator, idx_labels, valid_generator
