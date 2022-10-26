from re import X
from miniConvolutional.source.data import source_images
from miniConvolutional.model.encoder import build_encoder
from miniConvolutional.model.decoder import build_decoder
from miniConvolutional.model.autoencoder import build_autoencoder, compile_autoencoder

from miniConvolutional.model.loss import loss
from miniConvolutional.model.optimizer import optimizer
from miniConvolutional.model.metric import val_acc_metric, train_acc_metric
from miniConvolutional.model.convolutional import Convolutional

from colorama import Fore, Style
import os
import tensorflow as tf


# (train_images, train_labels), (test_images, test_labels) = source_images(mnist=True)

VALIDATION_STEPS, STEPS_PER_EPOCH, train_dataset, validation_dataset = source_images(
    cifar=True
)


if int(os.environ.get("AUTOENCODER")) == 1:

    X_train = train_images.reshape((60000, 28, 28, 1)) / 255.0

    X_test = test_images.reshape((10000, 28, 28, 1)) / 255.0

    latent_dim = 10

    encoder = build_encoder(latent_dim)

    decoder = build_decoder(latent_dim)

    autoencoder = build_autoencoder(encoder, decoder)

    print(encoder.summary())

    print(decoder.summary())

    print(autoencoder.summary())

    compile_autoencoder(autoencoder)

    autoencoder.fit(X_train, X_train, epochs=20, batch_size=32)

    # you can now display an image to see it is reconstructed well
    prediction = autoencoder.predict(X_train, verbose=0, batch_size=100)

# LABELS ARE FLATTEN
if int(os.environ.get("CLASSIFIER")) == 1:

    @tf.function
    def train_step(train_data, train_label):
        with tf.GradientTape() as tape:
            logits = model(train_data, training=True)

            print(
                "\nℹ️ "
                + Fore.CYAN
                + f"With train_label tensor {train_label}"
                + Style.RESET_ALL
            )

            print(
                "\nℹ️ " + Fore.CYAN + f"With logits tensor {logits}" + Style.RESET_ALL
            )

            loss_value = loss(train_label, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(train_label, logits)
        return loss_value

    @tf.function
    def test_step(validation_data, validation_label):
        val_logits = model(validation_data, training=False)
        val_acc_metric.update_state(validation_label, val_logits)

    import time

    # train_dataset = train_generator  # DirectoryIterator

    # train_dataset = train_images

    # validation_dataset = valid_generator  # DirectoryIterator

    # validation_dataset = test_images

    model = Convolutional(32)

    model.build((None, 32, 32, 3))

    print("\n⏹ " + Fore.BLUE + "The Model summary is" + Fore.YELLOW + "\n")

    print(model.summary())

    for epoch in range(int(os.environ.get("EPOCHS"))):

        print("\n⏩ " + Fore.RED + "Epoch number %d" % (epoch + 1,) + Style.RESET_ALL)

        start_time_epoch = time.time()

        # Iterate datasetr batches

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            start_time_step = time.time()

            loss_value = train_step(x_batch_train, y_batch_train)

            # results every 100
            if step % 100 == 0:

                print(
                    "\nℹ️ "
                    + Fore.RED
                    + "training loss (for one epoch) at step %d epoch %d: %.4f"
                    % (step, epoch, float(loss_value))
                    + Style.RESET_ALL
                )

            print(
                "\n📶 "
                + Fore.MAGENTA
                + "Sample procesed so far: %d samples"
                % ((step + 1) * int(os.environ.get("BATCH_SIZE")))
                + Style.RESET_ALL
            )

            # show accuracy at completed epoch
            train_accuracy = train_acc_metric.result()

            print(
                "\n📶 "
                + Fore.MAGENTA
                + "Training set accuracy is: %.4f for step %d in epoch %d"
                % (float(train_accuracy), step, epoch)
                + Style.RESET_ALL
            )

            # reset training metrics before next epoc *
            train_acc_metric.reset_state()

            for x_batch_val, y_batch_val in validation_dataset:
                test_step(x_batch_val, y_batch_val)

            val_accuracy = val_acc_metric.result()

            val_acc_metric.reset_state()

            print(
                "\n📶 "
                + Fore.MAGENTA
                + "Validation set accuracy is: %.4f for step %d in epoch %d"
                % (float(val_accuracy), step, epoch)
                + Style.RESET_ALL
            )

            print(
                "\nℹ️ "
                + Fore.CYAN
                + "Time taken for step %d: %.2fs"
                % (step, (time.time() - start_time_step))
                + Style.RESET_ALL
            )

        print(
            "\nℹ️ "
            + Fore.CYAN
            + "Time taken for epoch %d: %.2fs"
            % (epoch, (time.time() - start_time_epoch))
            + Style.RESET_ALL
        )

else:
    print("No model selected")
