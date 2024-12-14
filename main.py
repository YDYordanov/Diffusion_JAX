"""
This is an implementation of diffusion models in JAX
"""

from models import FFN
from utils import load_mnist_images, load_mnist_labels


def main():
    # Load the dataset
    data_folder = 'data/MNIST'
    x_train = load_mnist_images(file_name='train-images.idx3-ubyte', data_folder=data_folder)
    y_train = load_mnist_labels(file_name='train-labels.idx1-ubyte', data_folder=data_folder)
    print(x_train.shape, y_train.shape)

    # ToDo: do a train/dev split

    # ToDo: do batching (maybe via jax classes)

    x_test = load_mnist_images(file_name='t10k-images.idx3-ubyte', data_folder=data_folder)
    y_test = load_mnist_labels(file_name='t10k-labels.idx1-ubyte', data_folder=data_folder)

    # Number of pixels in each image
    in_size = x_train.shape[1] * x_train.shape[2]
    assert in_size == 784  # for MNIST
    print('Input size: {} pixels'.format(in_size))
    h_size = 128
    out_size = 10  # num classes

    model = FFN(in_size=in_size, h_size=h_size, out_size=out_size)

    model.run_epoch(x_data=x_train, y_data=y_train)
    model.evaluate(x_test_data=x_test, y_test_data=y_test)

if __name__ == "__main__":
    main()
