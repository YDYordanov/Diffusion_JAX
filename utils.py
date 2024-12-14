"""
Data and other utilities
"""
import os
import numpy as np


# Process the unzipped MNIST images
# Adapted from ChatGPT
def load_mnist_images(file_name, data_folder):
    filename = os.path.join(data_folder, file_name)
    with open(filename, 'rb') as f:
        # Read the header information
        magic_number, num_images, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32).byteswap()
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape to (num_images, rows, cols)
        images = images.reshape(num_images, rows, cols).astype(np.float32) / 255.0
        return images


# Process the unzipped MNIST labels
# Adapted from ChatGPT
def load_mnist_labels(file_name, data_folder):
    filename = os.path.join(data_folder, file_name)
    with open(filename, 'rb') as f:
        # Read the header information
        # magic_number, num_labels = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


class DataLoader:
    """
    The data loader does batching and random data shuffling
    """
    def __init__(self, data_array: np.array, b_size: int, do_shuffle=False, seed: int=10):
        self.b_size = b_size
        self.data_array = data_array

        # Cut the data into batches of size b_size
        # First, discard additional examples for simplicity
        num_extra_examples = self.data_array.shape[0] % self.b_size
        self.data_array = self.data_array[: -num_extra_examples]
        # Second, separate the batches
        num_examples = self.data_array.shape[0]
        examples_per_batch = num_examples // self.b_size
        print(self.data_array.shape)
        additional_dims = self.data_array.shape[1:]
        print(additional_dims)
        self.data_array = self.data_array.reshape(
            (self.b_size, examples_per_batch) + additional_dims)

        # Randomly shuffle the batches given a seed
        # ToDo: implement this

    def get_batch(self):
        pass
