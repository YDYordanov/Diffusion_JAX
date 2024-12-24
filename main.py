"""
This is an implementation of diffusion models in JAX
"""
import argparse
import time
import optax

from models import run_epoch, evaluate_model, ffn_jax
from utils import (load_mnist_data, load_cifar_data, random_train_dev_split,
                   DataLoader, inspect_data)
from jax import random
from jax.nn.initializers import glorot_normal


def main():
    # Enable command-line arguments
    parser = argparse.ArgumentParser()

    # The data arguments
    parser.add_argument(
        '-dataset', '--dataset_name', choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument(
        '-unflatten_images', '--unflatten_images', action='store_true',
        help='Use the images as multi-dimensional arrays, or flatten them.')
    parser.add_argument(
        '-inspect', '--inspect_data', action='store_true',
        help='Inspect a random sample of the data (images).')

    # The model and training parameters
    parser.add_argument(
        '-model', '--model_name', choices=['ffn'], default='ffn')
    parser.add_argument(
        '-bs', '--b_size', type=int, default=64, help='Training batch size.')
    parser.add_argument(
        '-hs', '--h_size', type=int, default=32, help='Model hidden size.')
    parser.add_argument(
        '-out_size', '--out_size', type=int, default=10,
        help='Number of classes = output size.')
    parser.add_argument(
        '-ep', '--num_epochs', type=int, default=1, help='Training epochs.')
    parser.add_argument(
        '-lr', '--lr', type=float, default=1e-3, help='Optimiser learning rate.')
    parser.add_argument(
        '-eval_int', '--eval_interval', type=int, default=10**5,
        help='Run dev evaluation every this many training steps.')
    parser.add_argument(
        '-test', '--do_test', action='store_true',
        help='Run test evaluation.')

    # Parse the arguments; access them via args.<argument>
    args = parser.parse_args()

    if args.unflatten_images:
        raise NotImplementedError('Models are not adapted for multi-dimensional images')

    assert args.eval_interval <= 2 ** 32  # jax.jit needs it to be int32

    # Load the dataset
    print('Loading and processing data...')
    if args.dataset_name == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist_data(
            data_folder='data/MNIST', use_flat_images=not args.unflatten_images)
    elif args.dataset_name == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar_data(
            data_folder='data/CIFAR-10', use_flat_images=not args.unflatten_images)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    else:
        raise NotImplementedError

    # Inspect the training data
    if args.inspect_data:
        inspect_data(
            dataset_name=args.dataset_name, x_data=x_train, y_data=y_train, sample_size=3)
        exit()

    # The input size (in_size) of the model equals the #pixels in each image
    if not args.unflatten_images:
        in_size = x_train.shape[1]
        if args.dataset_name == 'mnist':
            assert in_size == 784
            print('Input size: {} pixels'.format(in_size))
        elif args.dataset_name == 'cifar10':
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

    train_loader = DataLoader(x_data_array=x_train_new, y_data_array=y_train_new, b_size=args.b_size)
    dev_data_loader = DataLoader(x_data_array=x_dev, y_data_array=y_dev, b_size=args.b_size)
    test_data_loader = DataLoader(x_data_array=x_test,y_data_array=y_test, b_size=100)
    print('...the data is ready!')

    # Specify the model function
    if args.model_name == 'ffn':
        model_fn = ffn_jax
    else:
        raise NotImplementedError

    # Specify the parameter initialisation
    # Xavier initialisation
    init_fn = glorot_normal()
    params = {
        'layer1': {
            'W': init_fn(random.PRNGKey(12), (in_size, args.h_size)),
            'b': random.normal(random.PRNGKey(1322), (args.h_size))},
        'projection': {
            'W': init_fn(random.PRNGKey(23), (args.h_size, args.out_size)),
            'b': random.normal(random.PRNGKey(125), (args.out_size))}
    }

    # Create the optimiser and optimiser state
    optim = optax.adamw(learning_rate=args.lr)
    opt_state = optim.init(params)

    # Do training
    print('Training...')
    start_time = time.time()
    for epoch in range(args.num_epochs):
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
            num_classes=args.out_size,
            eval_interval=args.eval_interval
        )

        # Dev-evaluate the model
        dev_acc, dev_loss = evaluate_model(
            model_fn=model_fn,
            params=params,
            x_test_data=dev_data_loader.x_data_array,
            y_test_data=dev_data_loader.y_data_array,
            num_classes=args.out_size
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
        num_classes=args.out_size
    )
    print('Final dev accuracy:', dev_acc)
    print('Final dev loss:', dev_loss)

    if args.do_test:
        # Test-evaluate the final model
        evaluate_model(
            model_fn=model_fn,
            params=params,
            x_test_data=test_data_loader.x_data_array,
            y_test_data=test_data_loader.y_data_array,
            num_classes=args.out_size
        )

if __name__ == "__main__":
    main()
