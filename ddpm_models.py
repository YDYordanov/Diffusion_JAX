"""
Diffusion image model implemented from:
Denoising Diffusion Probabilistic Models (DDPM)
(https://arxiv.org/abs/2006.11239)
"""
import jax.numpy as jnp
import jax.random as jrand
import jax.nn as nn  # activation fn-s
from jax import grad
from tqdm import tqdm

import jax
import optax
import functools
import typing


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0))  # in_axes=(None, 0))
def ddpm_jax(params: dict, x: jnp.array):
    assert x.shape[0] == params['layer1']['W'].shape[0]

    # Forward pass through the network
    # ToDo: implement this?

    return x # out


@functools.partial(jax.jit, static_argnames=['model_fn'])
def compute_ddpm_loss(
        params: dict, x: jnp.array, y: jnp.array,
        model_fn: typing.Callable):
    # First, forward x through the network
    out = model_fn(params, x)

    # Compute the loss function used as training objective
    # ToDo: implement this
    # l = cross_entropy_loss(out, y, num_classes).sum(axis=-1).mean()

    return None  # l


@functools.partial(jax.jit, static_argnames=['model_fn', 'optim', 'T'])
def grad_and_update_ddpm(
        model_fn: typing.Callable, params: dict, optim, opt_state, x: jnp.array,
        T: int, seed: int):
    """
    This function implements Algorithm 1 from the DDPM work
    https://arxiv.org/abs/2006.11239
    """
    # We skip sampling x because we do it during data shuffling
    # Now, sample t and epsilon as t ~ Uniform({1, ..., T})
    # Get a uniform distribution from a categorical one with equal prob-s:
    uniform_logits = jnp.ones((T, x.shape[0]))
    t = jrand.categorical(key=jrand.key(seed=seed), logits=uniform_logits, axis=0) + 1

    # Then sample epsilon ~ N(0, I)
    # eps = jrand.multivariate_normal(key=jrand.key(seed=seed+1), mean=0, cov=, shape)

    # Compute the gradients
    # grads = grad(compute_ddpm_loss)(params, x, model_fn=model_fn, )

    # Update the optimiser and the parameters
    # ToDo: implement this
    # updates, opt_state = optim.update(updates=grads, state=opt_state, params=params)
    # params = optax.apply_updates(params, updates)
    return params, opt_state, t


def run_ddpm_epoch(
    model_fn: typing.Callable, params: dict, T: int, optim, opt_state,
    x_train_data: jnp.array, x_dev_data: jnp.array, eval_interval: int=10, seed: int=32598):
    """
    T: (int), the number of diffusion steps
    """
    for batch_id, x in tqdm(enumerate(x_train_data), total=x_train_data.shape[0]):

        # Use the data to compute the gradients and update the optimiser and the parameters
        # This is done in a separate function to enable jax.jit optimisation with compiling
        params, opt_state, t = grad_and_update_ddpm(
            model_fn=model_fn, params=params, optim=optim, opt_state=opt_state,
            x=jnp.array(x), T=T, seed=seed)

        if (batch_id + 1) % eval_interval == 0:
            # Dev evaluation
            # ToDo: change this
            dev_acc, dev_loss = evaluate_ddpm_model(model_fn, params, x_dev_data)
            print('Dev accuracy:', dev_acc)
            print('Dev loss:', dev_loss)

    return params, optim, opt_state


@functools.partial(jax.jit, static_argnames=['model_fn'])
def evaluate_ddpm_model(
        model_fn: typing.Callable, params: dict, x_test_data: jnp.array):
    """
    Return the accuracy w.r.t. y_test_data
    """
    # ToDo: implement this

    num_correct = 0
    num_all = 0
    num_batches = 0
    total_loss = 0
    for x in x_test_data:
        x = jnp.array(x)
        out = model_fn(params, x)
        # Model predictions
        predictions = out.argmax(axis=-1)
        num_correct += (predictions == y).sum()
        num_all += predictions.shape[0]
        num_batches += 1
        # total_loss += cross_entropy_loss(out, y, num_classes).sum(axis=-1).mean()
    # print('num correct/num all:', num_correct, num_all)
    return num_correct / num_all, total_loss / num_batches
