"""
Diffusion image model implemented from:
Denoising Diffusion Probabilistic Models (DDPM)
(https://arxiv.org/abs/2006.11239)
"""

import jax
import optax
import functools
import typing

import jax.numpy as jnp
import jax.random as jrand
import jax.nn as nn  # activation fn-s

from jax import grad
from tqdm import tqdm


def get_a_t_hat(T: int, b_1: float=1e-4, b_last: float=2e-2):
    """
    Compute a_t_hat and a_t from the DDPM work (https://arxiv.org/abs/2006.11239)
    :return: a_t_hat_values, a_t_values : arrays of values across t=1,...,T
    """
    # b_t (t=1,...,T) is a linear schedule from 10^-4 to 0.02
    b_t_values = jnp.array([b_1 + (b_last - b_1) * i / (T - 1) for i in range(0, T)])

    # a_t = 1 - b_t
    a_t_values = (1 - b_t_values)

    # a_t_hat := Product(a_i, i=1,...,t)
    a_t_hat_values = jnp.array([a_t_values[:t].prod() for t in range(1, T)])

    return a_t_hat_values, a_t_values


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0, 0))
def ddpm_ffn_model_fn(params: dict, x_noisy: jnp.array, t: jnp.array):
    """
    The epsilon_theta transformation of the noisy image and t,
    but simplified: using FFN instead of U-net transformer
    """
    assert x_noisy.shape[0] + 1 == params['layer1']['W'].shape[0]
    input_array = jnp.concatenate(
        arrays=(x_noisy, jnp.expand_dims(t, axis=0)), axis=0)

    # Forward pass through the network
    hid_state = jnp.matmul(input_array, params['layer1']['W']) + params['layer1']['b']
    hid_state = nn.gelu(hid_state)
    out = jnp.matmul(hid_state, params['projection']['W']) + params['projection']['b']

    return out


@functools.partial(jax.jit, static_argnames=['model_fn'])
def compute_ddpm_loss(
        params: dict, x: jnp.array, eps: jnp.array,
        t: jnp.array, a_t_hat_values: jnp.array, model_fn: typing.Callable):
    """
    Implementing the objective function of DDPM,
    provided in Algorithm 1 (https://arxiv.org/abs/2006.11239)
    """
    # First, forward x through the network
    # This corresponds to finding epsilon_theta in Algorthm 1
    a_t_hat = jnp.expand_dims(a_t_hat_values[t], axis=1)
    x_noisy = jnp.sqrt(a_t_hat * x) + jnp.sqrt(1 - a_t_hat) * eps
    eps_theta = model_fn(params, x_noisy, t)

    # Compute the MSE loss
    l = ((eps - eps_theta) ** 2).sum(axis=-1).mean()

    return l


@functools.partial(jax.jit, static_argnames=['model_fn', 'optim', 'T'])
def grad_and_update_ddpm(
        model_fn: typing.Callable, params: dict, optim, opt_state, x: jnp.array,
        T: int, a_t_hat_values: jnp.array, seed: int):
    """
    This function implements Algorithm 1 from the DDPM work
    https://arxiv.org/abs/2006.11239
    """
    # We skip sampling x because we do it during data shuffling
    # Now, sample t and epsilon as t ~ Uniform({1, ..., T})
    # Get a uniform distribution from a categorical one with equal prob-s:
    uniform_logits = jnp.ones((T, x.shape[0]))
    t = jrand.categorical(
        key=jrand.key(seed=seed), logits=uniform_logits, axis=0
    ) + 1

    # Then sample epsilon ~ N(0, I), of the same shape as x
    # Note: the sample is independent across both the batch and image dimensions
    eps = jrand.normal(key=jrand.key(seed=seed+325), shape=x.shape)

    # Compute the gradients
    grads = grad(compute_ddpm_loss)(
        params, x, eps, t, a_t_hat_values, model_fn=model_fn)

    # Update the optimiser and the parameters
    updates, opt_state = optim.update(updates=grads, state=opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, eps, t


def run_ddpm_epoch(
        model_fn: typing.Callable, params: dict, T: int, a_t_hat_values: jnp.array,
        optim, opt_state, x_train_data: jnp.array, x_dev_data: jnp.array,
        eval_interval: int=10, seed: int=32598):
    """
    T: (int), the number of diffusion steps
    """
    for batch_id, x in tqdm(enumerate(x_train_data), total=x_train_data.shape[0]):

        # Use the data to compute the gradients and update the optimiser and the parameters
        # This is done in a separate function to enable jax.jit optimisation with compiling
        params, opt_state, eps, t = grad_and_update_ddpm(
            model_fn=model_fn, params=params, a_t_hat_values=a_t_hat_values,
            optim=optim, opt_state=opt_state, x=jnp.array(x), T=T, seed=seed)

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
