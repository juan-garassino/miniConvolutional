from tensorflow.keras.losses import (
    SparseCategoricalCrossentropy,
    CategoricalCrossentropy,
)

loss = SparseCategoricalCrossentropy(from_logits=True)
