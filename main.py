"""
This is an implementation of diffusion models in JAX
"""
import os
import argparse
import time
import yaml
import optax
import ray

from functools import partial
from ray import tune
from ray.train import Checkpoint
from ray.tune.search.optuna import OptunaSearch

from models import run_epoch, evaluate_ffn_model, ffn_jax, ffn_init
from ddpm_models import (
    run_ddpm_epoch, sample_ddpm_image, ddpm_ffn_init, ddpm_ffn_model_fn,
    evaluate_ddpm_model, b_t_linear_schedule, a_t_hat_cosine_schedule)
from utils import (load_mnist_data, load_cifar_data, random_train_dev_split,
                   DataLoader, inspect_data, inspect_image,
                   save_checkpoint, load_checkpoint, str_to_number,
                   DotDict)


#@ray.remote(num_cpus=12, num_gpus=1)  # For creating multiple workers in parallel
def train_fn(config, model_constants, datasets_dict_ref, in_size, checkpoint_dir=None):
    """
    The main training function.
    This is needed for Ray Tune hyperparameter search.
    :config: a Ray-specific variable which provides the hyperparameter combination
    """
    # This handles large files being passed as arguments to train_fn
    datasets_dict = ray.get(datasets_dict_ref)
    assert type(datasets_dict) == dict

    config = {**config, **model_constants}

    # Enable access to config entries via config.<entry_name>
    config = DotDict(config)

    # Create the data loaders.
    # These are inside the train_fn() because they depend on the batch size, which is a hyperparameter
    train_loader = DataLoader(
        x_data_array=datasets_dict['x_train_new'], y_data_array=datasets_dict['y_train_new'], b_size=config.b_size)
    dev_data_loader = DataLoader(
        x_data_array=datasets_dict['x_dev'], y_data_array=datasets_dict['y_dev'], b_size=config.b_size)
    test_data_loader = DataLoader(
        x_data_array=datasets_dict['x_test'], y_data_array=datasets_dict['y_test'], b_size=100)
    print('...the data is ready!')

    # Specify the model function
    if config.model_name == 'ffn':
        model_fn = ffn_jax
    elif config.model_name == 'ddpm':
        model_fn = ddpm_ffn_model_fn
    else:
        raise NotImplementedError

    # Initialise the parameters
    print('\nInitialising model...')
    if config.model_name == 'ffn':
        params = ffn_init(
            num_h_layers=config.num_h_layers, in_size=in_size, h_size=config.h_size, out_size=config.num_classes)
    elif config.model_name == 'ddpm':
        # Same init as FFN, but larger in_size as x_noisy and t are concatenated as input;
        # the output is of the same dimensions as x.
        pos_emb_size = 128
        params = ddpm_ffn_init(
            num_h_layers=config.num_h_layers, in_size=in_size + pos_emb_size, h_size=config.h_size, out_size=in_size)
        # These are constants used for DDPM
        if config.diffusion_schedule == 'linear':
            a_t_hat_values, a_t_values = b_t_linear_schedule(b_1=1e-4, b_last=2e-2, T=config.T)
        elif config.diffusion_schedule == 'cosine':
            a_t_hat_values, a_t_values = a_t_hat_cosine_schedule(T=config.T)
    else:
        raise NotImplementedError

    print('\nInitialising optimiser...')
    # Create the optimiser and optimiser state
    optim = optax.adamw(learning_rate=config.lr)
    opt_state = optim.init(params)

    # Load the model and optimiser from checkpoint
    resume_epoch = 10e10  # placeholder value
    if config.resume_from_checkpoint is not None:
        params, opt_state, resume_epoch = load_checkpoint(
            checkpoint_dir=config.resume_from_checkpoint)

    # Do training
    print('\nTraining...')
    start_time = time.time()
    for epoch in range(config.num_epochs):
        # Shuffle the data at each epoch;
        # the first shuffle may be redundant
        data_shuffle_seed = 2304
        train_loader.do_shuffle(seed=data_shuffle_seed * (epoch + 1))

        # Skip epochs that are already trained
        if config.resume_from_checkpoint is not None:
            if epoch <= resume_epoch:
                continue

        # Run one epoch of training
        if config.model_name == 'ffn':
            # Train the FFN model for one epoch
            params, optim, opt_state = run_epoch(
                model_fn=model_fn, params=params,
                num_h_layers=config.num_h_layers,
                optim=optim, opt_state=opt_state,
                x_train_data=train_loader.x_data_array,
                y_train_data=train_loader.y_data_array,
                x_dev_data=dev_data_loader.x_data_array,
                y_dev_data=dev_data_loader.y_data_array,
                num_classes=config.num_classes,
                eval_interval=config.eval_interval
            )
        elif config.model_name == 'ddpm':
            # Train the DDPM model for one epoch
            epoch_seed = 250948 * (epoch + 1)
            params, optim, opt_state, train_loss = run_ddpm_epoch(
                model_fn=model_fn, params=params,
                num_h_layers=config.num_h_layers,
                T=config.T,
                a_t_values=a_t_values,
                a_t_hat_values=a_t_hat_values,
                optim=optim, opt_state=opt_state,
                x_train_data=train_loader.x_data_array,
                x_dev_data=dev_data_loader.x_data_array,
                eval_interval=config.eval_interval,
                seed=epoch_seed
            )
            print('Epoch {} train loss: {}'.format(epoch + 1, train_loss))

        # Dev-evaluate the model
        if config.model_name == 'ffn':
            dev_acc, dev_loss = evaluate_ffn_model(
                model_fn=model_fn,
                params=params,
                num_h_layers=config.num_h_layers,
                x_test_data=dev_data_loader.x_data_array,
                y_test_data=dev_data_loader.y_data_array,
                num_classes=config.num_classes
            )
            print('Epoch {}: dev accuracy: {}'.format(epoch + 1, dev_acc))
            print('Epoch {}: dev loss: {}'.format(epoch + 1, dev_loss))
        elif config.model_name == 'ddpm':
            print('\n---- Epoch {} dev evaluation ----'.format(epoch + 1))
            dev_loss_dict = evaluate_ddpm_model(
                model_fn=model_fn, params=params, num_h_layers=config.num_h_layers,
                a_t_values=a_t_values, a_t_hat_values=a_t_hat_values,
                x_test_data=dev_data_loader.x_data_array,
                T=config.T, verbose=True)
            dev_loss = dev_loss_dict['objective_loss']
        else:
            raise NotImplementedError

        # Save the model and optimiser checkpoint
        checkpoint_dir = config.save_dir
        save_checkpoint(
            params=params, opt_state=opt_state, epoch=epoch,
            checkpoint_dir=checkpoint_dir)
        """
        "In standard DDP training, where each worker has a copy of the full-model, 
        you should only save and report a checkpoint from a single worker to prevent redundant uploads."
        (https://docs.ray.io/en/latest/train/user-guides/checkpoints.html)
        Add: "if ray.train.get_context().get_world_rank() == 0:" for distributed data parallel (DDP) training
        """
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        metrics = {'dev_loss': dev_loss}
        ray.train.report(metrics=metrics, checkpoint=checkpoint) #, ray="tune")

    end_time = time.time()
    print('Training time:', end_time - start_time)

    # Dev-evaluate the final model
    print('\n---- Final dev evaluation ----')
    if config.model_name == 'ffn':
        dev_acc, dev_loss = evaluate_ffn_model(
            model_fn=model_fn,
            params=params,
            num_h_layers=config.num_h_layers,
            x_test_data=dev_data_loader.x_data_array,
            y_test_data=dev_data_loader.y_data_array,
            num_classes=config.num_classes
        )
        print('Final dev accuracy:', dev_acc)
        print('Final dev loss:', dev_loss)

    elif config.model_name == 'ddpm':
        dev_loss_dict = evaluate_ddpm_model(
            model_fn=model_fn, params=params, num_h_layers=config.num_h_layers,
            a_t_values=a_t_values, a_t_hat_values=a_t_hat_values,
            x_test_data=dev_data_loader.x_data_array,
            T=config.T)

        # Reconstruct a random image from noise and view it
        reconstructed_image_array = sample_ddpm_image(
            params=params, num_h_layers=config.num_h_layers,
            model_fn=model_fn, T=config.T, image_array_shape=(1, in_size),
            a_t_values=a_t_values, a_t_hat_values=a_t_hat_values,
            seed=epoch_seed + 10)
        inspect_image(dataset_name=config.dataset_name, image_array=reconstructed_image_array)

    if config.do_test:
        print('\n---- Test evaluation ----')
        if config.model_name == 'ffn':
            # Test-evaluate the final model
            test_acc, test_loss = evaluate_ffn_model(
                model_fn=model_fn,
                params=params,
                num_h_layers=config.num_h_layers,
                x_test_data=test_data_loader.x_data_array,
                y_test_data=test_data_loader.y_data_array,
                num_classes=config.num_classes
            )
            print('Final test accuracy:', test_acc)
            print('Final test loss:', test_loss)

        elif config.model_name == 'ddpm':
            test_loss_dict = evaluate_ddpm_model(
                model_fn=model_fn, params=params, num_h_layers=config.num_h_layers,
                a_t_values=a_t_values, a_t_hat_values=a_t_hat_values,
                x_test_data=test_data_loader.x_data_array,
                T=config.T)


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
    parser.add_argument(
        '-config', '--config_file', type=str, default='default_config.yaml',
        help='Set the configuration file specifying hyperparameter search and other configs.')

    parser.add_argument(
        '-save_dir', '--save_dir', type=str, default='saved_models/test',
        help='Directory to save the current run.')
    parser.add_argument(
        '-n_samples', '--num_samples', type=int, default=100,
        help='Number of sampled hyperparameter values when searching in Ray/Optuna.')

    # Parse the arguments; access them via args.<argument>
    args = parser.parse_args()

    if args.unflatten_images:
        raise NotImplementedError('Models are not adapted for multi-dimensional images')

    # Load the dataset
    print('\nLoading and processing data...')
    if args.dataset_name == 'mnist':
        x_train, y_train, x_test, y_test = load_mnist_data(
            data_folder='data/MNIST', use_flat_images=not args.unflatten_images)
    elif args.dataset_name == 'cifar10':
        x_train, y_train, x_test, y_test = load_cifar_data(
            data_folder='data/CIFAR-10', use_flat_images=not args.unflatten_images)
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

    ray.init(num_cpus=12, num_gpus=1)
    print('Available resources:', ray.available_resources())

    # Load the .yaml configuration file
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
        hyperparam_space = config['hyperparameter_space']
        print(hyperparam_space)
        model_constants = config['model_constants']
        print(model_constants)

    # jax.jit needs int variables to be at most int32
    assert float(hyperparam_space['eval_interval']['value']) <= 2 ** 32

    # Convert model constants to numbers if possible
    for const_name, const_value in model_constants.items():
        if model_constants[const_name] is not None:
            model_constants[const_name] = str_to_number(model_constants[const_name])

    # Convert the yaml configuration to a hyperparameter space for Ray Tune:
    # E.g. {"T": {"value": 1000}} becomes {"T": tune.choice([1000])}
    ray_hyperparam_space = {}
    for hyp_name, hyp_dict in hyperparam_space.items():
        # Only one hyperparameter rule per hyperparameter
        hyp_rules = list(hyp_dict.keys())
        assert len(hyp_rules) == 1
        hyp_rule = hyp_rules[0]

        hyp_values = hyp_dict[hyp_rule]
        # The hyperparameter values have to always be a list
        if not isinstance(hyp_values, list):
            hyp_values = [hyp_values]
        for i in range(len(hyp_values)):
            # Check if the string is a number and convert to the appropriate number type
            if isinstance(hyp_values[i], str):
                # Convert the string to a number if possible
                hyp_values[i] = str_to_number(hyp_values[i])

        if hyp_rule in ['value', 'choice']:
            # a single value or a choice of values, preferably not numbers;
            # choice of numbers should be instead handled by uniform and loguniform if possible
            ray_hyperparam_space[hyp_name] = tune.choice(hyp_values)
        # Uniform and loguniform are preferred for advanced searchers; avoid choices
        elif hyp_rule == 'uniform':
            assert len(hyp_values) == 2
            ray_hyperparam_space[hyp_name] = tune.uniform(hyp_values[0], hyp_values[1])
        elif hyp_rule == 'loguniform':
            assert len(hyp_values) == 2
            ray_hyperparam_space[hyp_name] = tune.loguniform(hyp_values[0], hyp_values[1])
        else:
            raise NotImplementedError

    datasets_dict = {
        'x_train_new': x_train_new,
        'y_train_new': y_train_new,
        'x_dev': x_dev,
        'y_dev': y_dev,
        'x_test': x_test,
        'y_test': y_test
    }
    # Properly pass large objects to the train_fn()
    # (suggested by ChatGPT; large files as arguments lead to errors with Ray workers)
    datasets_dict_ref = ray.put(datasets_dict)

    # Pass additional arguments to train_fn() for ray Tuner
    ray_train_fn = partial(
        train_fn,
        model_constants=model_constants, in_size=in_size,
        datasets_dict_ref=datasets_dict_ref
    )

    # Allocate resources
    ray_train_fn = tune.with_resources(ray_train_fn, resources={"cpu": 12, "gpu": 1})

    tuner = tune.Tuner(
        ray_train_fn,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(),
            num_samples=args.num_samples,
            metric="dev_loss",
            mode="min",
        ),
        param_space=ray_hyperparam_space,
    )
    results = tuner.fit()

if __name__ == "__main__":
    main()
