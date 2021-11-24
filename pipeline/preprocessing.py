import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    """Preprocesses Covertype Dataset."""

    outputs = {}
    print("called")
    # print(inputs)
    return inputs