"""
This is an implementation of diffusion models in JAX
"""

from models import FFN
from utils import (load_mnist_images, load_mnist_labels, random_train_dev_split,
                   DataLoader)


def main():
    # Load the dataset
    data_folder = 'data/MNIST'
    x_train = load_mnist_images(file_name='train-images.idx3-ubyte', data_folder=data_folder)
    y_train = load_mnist_labels(file_name='train-labels.idx1-ubyte', data_folder=data_folder)
    x_test = load_mnist_images(file_name='t10k-images.idx3-ubyte', data_folder=data_folder)
    y_test = load_mnist_labels(file_name='t10k-labels.idx1-ubyte', data_folder=data_folder)

    # Set training parameters
    b_size = 16
    h_size = 128
    out_size = 10  # num classes
    # Number of pixels in each image
    in_size = x_train.shape[1] * x_train.shape[2]
    assert in_size == 784  # for MNIST
    print('Input size: {} pixels'.format(in_size))

    # Do a random train/dev split
    dev_proportion = 0.1
    random_split_seed = 235790
    x_train_new, y_train_new, x_dev, y_dev = random_train_dev_split(
        x_data=x_train, y_data=y_train, dev_proportion=dev_proportion, seed=random_split_seed
    )

    train_loader = DataLoader(x_data_array=x_train_new, y_data_array=y_train_new, b_size=b_size)
    dev_data_loader = DataLoader(x_data_array=x_dev, y_data_array=y_dev, b_size=b_size)
    print(x_test.shape, y_test.shape)
    test_data_loader = DataLoader(x_data_array=x_test,y_data_array=y_test, b_size=100)

    model = FFN(in_size=in_size, h_size=h_size, out_size=out_size)

    for epoch in range(1):
        # Shuffle the data at each epoch;
        # the first shuffle may be redundant
        data_shuffle_seed = 2304
        train_loader.do_shuffle(seed=data_shuffle_seed * (epoch + 1))
        model.run_epoch(
            x_train_data=train_loader.x_data_array,
            y_train_data=train_loader.y_data_array,
            x_dev_data=dev_data_loader.x_data_array,
            y_dev_data=dev_data_loader.y_data_array)
    model.evaluate(
        x_test_data=test_data_loader.x_data_array,
        y_test_data=test_data_loader.y_data_array)

if __name__ == "__main__":
    main()
