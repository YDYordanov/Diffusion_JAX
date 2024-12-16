"""
This is an implementation of diffusion models in JAX
"""

from models import run_epoch, evaluate_model, ffn_jax
from utils import (process_mnist, random_train_dev_split,
                   DataLoader, print_image)
from jax import random

import optax


def main():
    # Load the dataset
    data_folder = 'data/MNIST'
    x_train = process_mnist(file_name='train-images.idx3-ubyte', data_folder=data_folder)
    y_train = process_mnist(file_name='train-labels.idx1-ubyte', data_folder=data_folder)
    x_test = process_mnist(file_name='t10k-images.idx3-ubyte', data_folder=data_folder)
    y_test = process_mnist(file_name='t10k-labels.idx1-ubyte', data_folder=data_folder)

    # Inspect the data
    #data_id = 45972
    #print('image id:', y_train[data_id])
    #print_image(x_train[data_id])
    #data_id = 2359
    #print('image id:', y_test[data_id])
    #print_image(x_test[data_id])

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

    # Initialise the parameters
    params = {
        'layer1': {
            'W': random.normal(random.PRNGKey(12), (in_size, h_size)),
            'b': random.normal(random.PRNGKey(1322), (h_size))},
        'projection': {
            'W': random.normal(random.PRNGKey(23), (h_size, 1)),
            'b': random.normal(random.PRNGKey(125), (out_size))}
    }

    lr = 1e-3
    # Create the optimiser and optimiser state
    optim = optax.adamw(learning_rate=lr)
    opt_state = optim.init(params)

    model = FFN(in_size=in_size, h_size=h_size, out_size=out_size)

    for epoch in range(1):
        # Shuffle the data at each epoch;
        # the first shuffle may be redundant
        data_shuffle_seed = 2304
        train_loader.do_shuffle(seed=data_shuffle_seed * (epoch + 1))
        params, optim, opt_state = run_epoch(
            model_fn=ffn_jax, params=params,
            optim=optim, opt_state=opt_state,
            x_train_data=train_loader.x_data_array,
            y_train_data=train_loader.y_data_array,
            x_dev_data=dev_data_loader.x_data_array,
            y_dev_data=dev_data_loader.y_data_array,
            num_classes=out_size)
    evaluate_model(
        model_fn=ffn_jax,
        params=params,
        x_test_data=test_data_loader.x_data_array,
        y_test_data=test_data_loader.y_data_array)

if __name__ == "__main__":
    main()
