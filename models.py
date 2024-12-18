"""
This is an implementation of diffusion models in JAX
"""

import jax.numpy as jnp
import jax.nn as nn  # activation fn-s
from jax import grad
from tqdm import tqdm

import jax
import optax
import functools
import typing


@jax.jit
@functools.partial(jax.vmap, in_axes=(None, 0))  # in_axes=(None, 0))
def ffn_jax(params: dict, x: jnp.array):
    assert x.shape[0] * x.shape[1] == params['layer1']['W'].shape[0]
    x = x.reshape(-1)  # flatten the image dimensions

    # Forward pass through the network
    hid_state = jnp.matmul(x, params['layer1']['W']) + params['layer1']['b']
    hid_state = nn.gelu(hid_state)
    out = jnp.matmul(hid_state, params['projection']['W']) + params['projection']['b']

    return out


@functools.partial(jax.jit, static_argnames=['num_classes'])  # needs num_classes to be static
@functools.partial(jax.vmap, in_axes=(0, 0, None))
def cross_entropy_loss(out: jnp.array, y: jnp.array, num_classes: int=10):
    # First, compute softmax of the outputs to obtain probabilities
    out_probs = nn.softmax(out, axis=-1)

    # 1-hot encode the targets y
    y_vector = nn.one_hot(y, num_classes=num_classes)

    # Compute log-likelihood loss w.r.t. y_vector
    l = - jnp.log(out_probs) * y_vector
    # Note: we skipped ".sum(axis=-1).mean()" in order to apply jax.vmap vectorisation and batching

    return l


@functools.partial(jax.jit, static_argnames=['model_name', 'num_classes'])
def compute_ffn_loss(
        params: dict, x: jnp.array, y: jnp.array,
        model_name: str, num_classes: int=10):
    if model_name == 'ffn':
        model_fn = ffn_jax
    else:
        raise NotImplementedError
    # First, forward x through the network
    out = model_fn(params, x)

    # Compute the loss of output w.r.t. target y
    l = cross_entropy_loss(out, y, num_classes).sum(axis=-1).mean()

    return l


@functools.partial(jax.jit, static_argnames=['model_name', 'num_classes', 'optim'])
def grad_and_update(model_name: str, params: dict, optim, opt_state,
                    x: jnp.array, y: jnp.array, num_classes: int):

    # Compute the gradients
    grads = grad(compute_ffn_loss)(params, x, y, model_name=model_name, num_classes=num_classes)

    # Update the optimiser and the parameters
    updates, opt_state = optim.update(updates=grads, state=opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def run_epoch(
        model_name: str, params: dict, optim, opt_state, x_train_data: jnp.array, y_train_data: jnp.array,
        x_dev_data: jnp.array, y_dev_data: jnp.array, num_classes: int=10, eval_interval: int=10):
    for batch_id, (x, y) in tqdm(enumerate(zip(x_train_data, y_train_data)), total=x_train_data.shape[0]):

        # Compute the gradients and update the optimiser and the parameters
        # This is done in a separate function to enable jax.jit optimisation with compiling
        params, opt_state = grad_and_update(
            model_name=model_name, params=params, optim=optim, opt_state=opt_state,
            x=x, y=y, num_classes=num_classes)

        if (batch_id + 1) % eval_interval == 0:
            # Dev evaluation
            dev_acc, dev_loss = evaluate_model(model_name, params, x_dev_data, y_dev_data, num_classes)
            print('Dev accuracy:', dev_acc)
            print('Dev loss:', dev_loss)

    return params, optim, opt_state


@functools.partial(jax.jit, static_argnames=['model_name', 'num_classes'])
def evaluate_model(
        model_name: str, params: dict, x_test_data: jnp.array, y_test_data: jnp.array,
        num_classes: int):
    """
    Return the accuracy w.r.t. y_test_data
    """
    if model_name == 'ffn':
        model_fn = ffn_jax
    else:
        raise NotImplementedError

    num_correct = 0
    num_all = 0
    num_batches = 0
    total_loss = 0
    for x, y in zip(x_test_data, y_test_data):
        out = model_fn(params, x)
        # Model predictions
        predictions = out.argmax(axis=-1)
        num_correct += (predictions == y).sum()
        num_all += predictions.shape[0]
        num_batches += 1
        total_loss += cross_entropy_loss(out, y, num_classes).sum(axis=-1).mean()
    print('num correct/num all:', num_correct, num_all)
    return num_correct / num_all, total_loss / num_batches
