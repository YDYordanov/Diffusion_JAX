"""
Data and other utilities
"""
import os
import jax.random as jrand
import jax.numpy as jnp
import numpy as np


# Process the unzipped MNIST images
# Adapted from ChatGPT
def load_mnist_images(file_name: str, data_folder: str):
    filename = os.path.join(data_folder, file_name)
    with open(filename, 'rb') as f:
        # Read the header information
        magic_number, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape to (num_images, rows, cols)
        images = images.reshape(num_images, rows, cols).astype(np.float32) / 255.0
        return jnp.array(images)


# Process the unzipped MNIST labels
# Adapted from ChatGPT
def load_mnist_labels(file_name: str, data_folder: str):
    filename = os.path.join(data_folder, file_name)
    with open(filename, 'rb') as f:
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return jnp.array(labels)


def random_train_dev_split(x_data, y_data, dev_proportion, seed: int=10):
    # First, shuffle the data
    key = jrand.PRNGKey(seed)
    id_permutation = jrand.permutation(key, jnp.arange(x_data.shape[0]), axis=0)
    x_data = x_data[id_permutation]
    y_data = y_data[id_permutation]

    # Then, split the data according to dev_proportion
    dev_size = int(x_data.shape[0] * dev_proportion)
    x_train_data = x_data[dev_size:]
    x_dev_data = x_data[:dev_size]
    y_train_data = y_data[dev_size:]
    y_dev_data = y_data[:dev_size]

    return x_train_data, y_train_data, x_dev_data, y_dev_data


class DataLoader:
    """
    The data loader does batching and random data shuffling
    """
    def __init__(self, data_array: jnp.array, b_size: int, seed: int=10):
        self.b_size = b_size
        self.data_array = data_array

        # Cut the data into batches of size b_size
        # First, discard additional examples for simplicity
        num_extra_examples = self.data_array.shape[0] % self.b_size
        if num_extra_examples != 0:
            self.data_array = self.data_array[: -num_extra_examples]
        # Second, separate the batches
        num_examples = self.data_array.shape[0]
        num_batches = num_examples // self.b_size
        additional_dims = self.data_array.shape[1:]
        new_shape = (num_batches, self.b_size) + additional_dims
        self.data_array = self.data_array.reshape(new_shape)

        # Convert to JAX array, sending it to the designated device
        #self.data_array = jnp.array(self.data_array)

        print('Data array shape:', self.data_array.shape)


    def do_shuffle(self, seed: int=10):
        # Randomly shuffle the batches given a seed
        key = jrand.PRNGKey(seed)
        self.data_array = jrand.permutation(key, self.data_array, axis=0)
