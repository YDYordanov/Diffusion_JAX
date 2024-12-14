"""
This is an implementation of diffusion models in JAX
"""

from models import FFN
from utils import load_mnist_images, load_mnist_labels, DataLoader


def main():
    b_size = 16

    # Load the dataset
    # ToDo: do a train/dev split
    data_folder = 'data/MNIST'
    x_train = load_mnist_images(file_name='train-images.idx3-ubyte', data_folder=data_folder)
    y_train = load_mnist_labels(file_name='train-labels.idx1-ubyte', data_folder=data_folder)
    x_train_loader = DataLoader(data_array=x_train, b_size=b_size, seed=2309, do_shuffle=True)
    y_train_loader = DataLoader(data_array=y_train, b_size=b_size, seed=2085, do_shuffle=True)

    x_test = load_mnist_images(file_name='t10k-images.idx3-ubyte', data_folder=data_folder)
    y_test = load_mnist_labels(file_name='t10k-labels.idx1-ubyte', data_folder=data_folder)
    x_test_loader = DataLoader(data_array=x_test, b_size=100)
    y_test_loader = DataLoader(data_array=y_test, b_size=100)

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
