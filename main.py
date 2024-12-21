"""
This is an implementation of diffusion models in JAX
"""

from models import run_epoch, evaluate_model, ffn_jax
from utils import (load_mnist_data, load_cifar_data, random_train_dev_split,
                   DataLoader, print_image)
from jax import random
from jax.nn.initializers import glorot_normal

import optax
import time


def main():
    dataset = 'cifar10'
    assert dataset in ['mnist', 'cifar10']
    use_flat_images = True  # to flatten the image dimensions or not
    if not use_flat_images:
        raise NotImplementedError

    # The model and training parameters
    model_name = 'ffn'
    b_size = 64
    h_size = 32
    out_size = 10  # num classes
    num_epochs = 1
    lr = 1e-3
    eval_interval = 10 ** 5
    assert eval_interval <= 2 ** 32  # jax.jit needs it to be int32
    do_test = False  # test evaluation

    # Load the dataset
    print('Loading and processing data...')
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist_data(
            data_folder='data/MNIST', use_flat_images=use_flat_images)
    elif dataset == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar_data(
            data_folder='data/CIFAR-10', use_flat_images=use_flat_images)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    else:
        raise NotImplementedError

    # Inspect the data
    #data_id = 45972
    #print('image id:', y_train[data_id])
    #print_image(x_train[data_id])
    #data_id = 2359
    #print('image id:', y_test[data_id])
    #print_image(x_test[data_id])

    # The input size (in_size) of the model equals the #pixels in each image
    if use_flat_images:
        in_size = x_train.shape[1]
        if dataset == 'mnist':
            assert in_size == 784
            print('Input size: {} pixels'.format(in_size))
        elif dataset == 'cifar10':
            assert in_size == 3072
            print('Input size: {} pixels with colours'.format(in_size))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # Do a random train/dev split
    dev_proportion = 0.05
    random_split_seed = 235790
    x_train_new, y_train_new, x_dev, y_dev = random_train_dev_split(
        x_data=x_train, y_data=y_train, dev_proportion=dev_proportion, seed=random_split_seed
    )

    train_loader = DataLoader(x_data_array=x_train_new, y_data_array=y_train_new, b_size=b_size)
    dev_data_loader = DataLoader(x_data_array=x_dev, y_data_array=y_dev, b_size=b_size)
    test_data_loader = DataLoader(x_data_array=x_test,y_data_array=y_test, b_size=100)
    print('...the data is ready!')

    # Specify the model function
    if model_name == 'ffn':
        model_fn = ffn_jax
    else:
        raise NotImplementedError

    # Specify the parameter initialisation
    # Xavier initialisation
    init_fn = glorot_normal()
    params = {
        'layer1': {
            'W': init_fn(random.PRNGKey(12), (in_size, h_size)),
            'b': random.normal(random.PRNGKey(1322), (h_size))},
        'projection': {
            'W': init_fn(random.PRNGKey(23), (h_size, out_size)),
            'b': random.normal(random.PRNGKey(125), (out_size))}
    }

    # Create the optimiser and optimiser state
    optim = optax.adamw(learning_rate=lr)
    opt_state = optim.init(params)

    # Do training
    print('Training...')
    start_time = time.time()
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch;
        # the first shuffle may be redundant
        data_shuffle_seed = 2304
        train_loader.do_shuffle(seed=data_shuffle_seed * (epoch + 1))

        # Run one epoch of training
        params, optim, opt_state = run_epoch(
            model_fn=model_fn, params=params,
            optim=optim, opt_state=opt_state,
            x_train_data=train_loader.x_data_array,
            y_train_data=train_loader.y_data_array,
            x_dev_data=dev_data_loader.x_data_array,
            y_dev_data=dev_data_loader.y_data_array,
            num_classes=out_size,
            eval_interval=eval_interval
        )

        # Dev-evaluate the model
        dev_acc, dev_loss = evaluate_model(
            model_fn=model_fn,
            params=params,
            x_test_data=dev_data_loader.x_data_array,
            y_test_data=dev_data_loader.y_data_array,
            num_classes=out_size
        )
        print('Epoch {} dev accuracy: {}'.format(epoch+1, dev_acc))
        print('Epoch {} dev loss: {}'.format(epoch+1, dev_loss))

    end_time = time.time()
    print('Training time:', end_time - start_time)

    # Dev-evaluate the final model
    dev_acc, dev_loss = evaluate_model(
        model_fn=model_fn,
        params=params,
        x_test_data=dev_data_loader.x_data_array,
        y_test_data=dev_data_loader.y_data_array,
        num_classes=out_size
    )
    print('Final dev accuracy:', dev_acc)
    print('Final dev loss:', dev_loss)

    if do_test:
        # Test-evaluate the final model
        evaluate_model(
            model_fn=model_fn,
            params=params,
            x_test_data=test_data_loader.x_data_array,
            y_test_data=test_data_loader.y_data_array,
            num_classes=out_size
        )

if __name__ == "__main__":
    main()
