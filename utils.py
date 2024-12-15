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


def joint_shuffle(x: jnp.array, y: jnp.array, seed: int=10, axis: int=0):
    """
    jointly shuffle two JAX arrays across an axis
    """
    key = jrand.PRNGKey(seed)
    id_permutation = jrand.permutation(key, jnp.arange(x.shape[0]), axis=axis)
    x_shuffled = x[id_permutation]
    y_shuffled = y[id_permutation]
    return x_shuffled, y_shuffled


def random_train_dev_split(x_data, y_data, dev_proportion, seed: int=10):
    # First, shuffle the data
    x_data, y_data = joint_shuffle(x_data, y_data, seed=seed, axis=0)

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
    def __init__(self, x_data_array: jnp.array, y_data_array: jnp.array, b_size: int):
        self.b_size = b_size
        print(x_data_array.shape, y_data_array.shape)
        assert x_data_array.shape[0] == y_data_array.shape[0]
        self.unbatched_x_data_array = x_data_array
        self.unbatched_y_data_array = y_data_array

        self.num_examples = self.unbatched_x_data_array.shape[0]
        self.num_batches = self.num_examples // self.b_size
        self.num_extra_examples = self.num_examples % self.b_size

        self.x_data_array = self.cut_and_batch_data(
            data_array=self.unbatched_x_data_array, num_batches=self.num_batches,
            num_examples_to_drop=self.num_extra_examples)
        self.y_data_array = self.cut_and_batch_data(
            data_array=self.unbatched_y_data_array, num_batches=self.num_batches,
            num_examples_to_drop=self.num_extra_examples)

        # Convert to JAX array, sending it to the designated device
        #self.data_array = jnp.array(self.data_array)

        print('Data array shapes:', self.x_data_array.shape, self.y_data_array.shape)


    def cut_and_batch_data(self, data_array, num_batches, num_examples_to_drop):
        # First, discard additional examples for simplicity
        if num_examples_to_drop != 0:
            data_array = data_array[: -num_examples_to_drop]

        # Second, separate the batches
        additional_dims = data_array.shape[1:]
        batched_shape = (num_batches, self.b_size) + additional_dims
        batched_data_array = data_array.reshape(batched_shape)

        return batched_data_array


    def do_shuffle(self, seed: int=10):
        # Randomly jointly shuffle the x and y raw data, given a seed
        self.unbatched_x_data_array, self.unbatched_y_data_array = joint_shuffle(
            x=self.unbatched_x_data_array, y=self.unbatched_y_data_array,
            seed=seed, axis=0
        )

        # Cut and batch data
        # Note: shuffling is before batching, for more randomness
        self.x_data_array = self.cut_and_batch_data(
            data_array=self.unbatched_x_data_array, num_batches=self.num_batches,
            num_examples_to_drop=self.num_extra_examples)
        self.y_data_array = self.cut_and_batch_data(
            data_array=self.unbatched_y_data_array, num_batches=self.num_batches,
            num_examples_to_drop=self.num_extra_examples)
