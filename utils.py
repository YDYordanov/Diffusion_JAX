"""
Data and other utilities
"""
import os
import struct
import pickle
import random

import jax.random as jrand
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np



def print_image(image_array: np.array):
    """
    Display an image from a 2-D array
    """
    assert len(image_array.shape) == 2

    # Adapted from ChatGPT
    # Rescale pixel values from [-1, 1] to [0, 1]:
    rescaled_image = (np.array(image_array) + 1) / 2
    plt.imshow(rescaled_image, cmap='gray')
    plt.axis('off')  # Turn off axes for a cleaner look
    plt.show()


def print_colour_image(image_array: np.array):
    """
    Display an image from a 3-D array:
    "(M, N, 3): an image with RGB values (0-1 float or 0-255 int)." (imshow docs)
    """
    assert len(image_array.shape) == 3 and image_array.shape[2] == 3

    # Plot the image
    # Rescale pixel values from [-1, 1] to [0, 1]:
    rescaled_image = (np.array(image_array) + 1) / 2
    plt.imshow(rescaled_image)
    plt.axis('off')  # Turn off axes for a cleaner look
    plt.show()


def process_mnist(file_name: str, data_folder: str, flatten_images: bool=False):
    """
    Process the unzipped MNIST images
    Reads an IDX file and returns the data as a NumPy array.
    (adapted from ChatGPT)

    :param data_folder: the folder containing the data files
    :param file_name: open the file in data_folder
    :param flatten_images: flatten the image dimensions or not
    """
    file_path = os.path.join(data_folder, file_name)
    with open(file_path, 'rb') as f:
        # Read the magic number
        magic = struct.unpack('>I', f.read(4))[0]
        # Read the number of items (labels or images)
        num_items = struct.unpack('>I', f.read(4))[0]

        if magic == 2049:  # Labels file
            # Read labels (unsigned bytes)
            data = np.frombuffer(f.read(), dtype=np.uint8)
        elif magic == 2051:  # Images file
            # Read dimensions
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            # Read image data (unsigned bytes)
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows, cols)
            data = (data / 255.0) * 2 - 1  # scaled to [0, 1], then to [-1, 1], for diffusion
            # Flatten the image dimensions
            if flatten_images:
                data = data.reshape(num_items, -1)
        else:
            raise ValueError(f"Invalid IDX file: magic number {magic}")

    return data


def load_mnist_data(data_folder: str, use_flat_images: bool=False):
    """ Load all MNIST data """
    x_train = process_mnist(
        file_name='train-images.idx3-ubyte', data_folder=data_folder, flatten_images=use_flat_images)
    y_train = process_mnist(
        file_name='train-labels.idx1-ubyte', data_folder=data_folder, flatten_images=use_flat_images)
    x_test = process_mnist(
        file_name='t10k-images.idx3-ubyte', data_folder=data_folder, flatten_images=use_flat_images)
    y_test = process_mnist(
        file_name='t10k-labels.idx1-ubyte', data_folder=data_folder, flatten_images=use_flat_images)

    return x_train, y_train, x_test, y_test


def unflatten_mnist_image(flat_image_array: np.array):
    """
    Unflatten one MNIST images: 2D images

    Args:
        flat_image_array (np.array): Flat image data of shape (784).

    Returns:
        np.array: Restored image array of shape (28, 28).
    """

    # Reshape to (num_images, 3, 32, 32)
    reshaped_image_array = flat_image_array.reshape(28, 28)

    return reshaped_image_array


def unpickle(file):
    """
    Load a pickled CIFAR-10 batch file.
    Instructions: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def unflatten_cifar_images(flat_image_data: np.array):
    """
    Unflatten the CIFAR images: separate the rgb channels and image dimensions
    (adapted from ChatGPT)

    Args:
        flat_image_data (np.array): Flat image data of shape (num_images, 3072).

    Returns:
        np.array: Restored image data of shape (num_images, 32, 32, 3).
    """

    # Number of images
    num_images = flat_image_data.shape[0]

    # Reshape to (num_images, 3, 32, 32)
    reshaped_data = flat_image_data.reshape(num_images, 3, 32, 32)

    # Transpose to (num_images, 32, 32, 3) to match image format
    restored_image_data = reshaped_data.transpose(0, 2, 3, 1)

    return restored_image_data


def load_cifar_data(data_folder: str, use_flat_images: bool=False):
    """
    Load and process the CIFAR-10 training and test data
    (adapted from ChatGPT)
    """
    # Paths to batch files
    batch_files = [f"{data_folder}/data_batch_{i}" for i in range(1, 6)]
    test_file = f"{data_folder}/test_batch"

    # Load and combine training batches
    train_images = []
    train_labels = []
    for batch_file in batch_files:
        batch = unpickle(batch_file)
        train_images.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    train_images = np.vstack(train_images)  # Stack arrays vertically
    train_labels = np.array(train_labels)

    # Load test batch
    test_batch = unpickle(test_file)
    test_images = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])

    # Unflatten the images if necessary:
    if not use_flat_images:
        train_images = unflatten_cifar_images(train_images)
        test_images = unflatten_cifar_images(test_images)

    # Convert to float32 to save memory, then normalise to [-1, 1] for diffusion
    train_images = (train_images.astype('float32') / 255.0) * 2 - 1
    test_images = (test_images.astype('float32') / 255.0) * 2 - 1

    return train_images, train_labels, test_images, test_labels


def inspect_data(
        dataset_name: str, x_data: np.array, y_data: np.array,
        sample_size: int=3):
    """
    Print a few random images from the dataset

    :param dataset_name: cifar10/mnist
    :param x_data: image array
    :param y_data: label array
    :param sample_size: number of images to print
    """
    # pick a few random images
    data_ids = random.sample(range(len(x_data)), sample_size)

    # Print each image
    for data_id in data_ids:
        if dataset_name == 'cifar10':
            classes = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
            print('image label:', classes[y_data[data_id]])
            sample_image = unflatten_cifar_images(x_data[data_id: data_id + 1])[0]
            print_colour_image(sample_image)

        elif dataset_name == 'mnist':
            print('image digit:', y_data[data_id])
            image_array = x_data[data_id]
            # Expand the image
            image_array = unflatten_mnist_image(image_array)
            print_image(image_array)


def joint_shuffle(x: jnp.array, y: jnp.array, seed: int=10, axis: int=0):
    """
    jointly shuffle two JAX arrays across an axis
    """
    key = jrand.PRNGKey(seed)
    id_permutation = jrand.permutation(key, x.shape[axis], axis=axis)
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
        assert x_data_array.shape[0] == y_data_array.shape[0]
        self.unbatched_x_data_array = x_data_array
        self.unbatched_y_data_array = y_data_array
        #array1 = jax.device_put(x_data_array, device=jax.devices("cpu")[0])

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

        print('Batched data array shapes:', self.x_data_array.shape, self.y_data_array.shape)


    def cut_and_batch_data(self, data_array, num_batches, num_examples_to_drop):
        # First, discard additional examples for simplicity
        # Note: for test datasets one should consider keeping the extra examples,
        # by having a smaller final batch.
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
