"""
Data and other utilities
"""
import os
import numpy as np


# Process unzipped MNIST images
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


# Process unzipped MNIST labels
# Adapted from ChatGPT
def load_mnist_labels(file_name, data_folder):
    filename = os.path.join(data_folder, file_name)
    with open(filename, 'rb') as f:
        # Read the header information
        # magic_number, num_labels = np.frombuffer(f.read(8), dtype=np.uint32).byteswap()
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels
