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

from jax import value_and_grad
from tqdm import tqdm


def a_t_hat_cosine_schedule(T: int, s=0.008):
    """
    Compute a_t_hat and a_t via cosine schedule,
    from the IDDPM work (https://arxiv.org/abs/2102.09672)
    :return: a_t_hat_values, a_t_values : arrays of values across t=1,...,T
    """
    # The cosine schedule:
    a_t_hat_values = jnp.array([
        jnp.cos(((t / T + s) / (1 + s)) * (jnp.pi/2)) ** 2
        / jnp.cos((s / (1 + s)) * (jnp.pi/2)) ** 2 for t in range(1, T+1)])

    # The reverse formula to obtain b_t
    b_t_values = jnp.minimum(
        jnp.array(
            [1 - a_t_hat_values[t - 1] / a_t_hat_values[t - 2] if t > 1 else 1 - a_t_hat_values[0]
            for t in range(1, T+1)]),
        0.999)

    # Get a_t from b_t, as usual
    a_t_values = (1 - b_t_values)

    return a_t_hat_values, a_t_values


def b_t_linear_schedule(T: int, b_1: float=1e-4, b_last: float=2e-2):
    """
    Compute a_t_hat and a_t from the DDPM work (https://arxiv.org/abs/2006.11239)
    :return: a_t_hat_values, a_t_values : arrays of values across t=1,...,T
    """
    # b_t (t=1,...,T) is a linear schedule from 10^-4 to 0.02
    b_t_values = jnp.array([b_1 + (b_last - b_1) * i / (T - 1) for i in range(0, T)])

    # a_t = 1 - b_t
    a_t_values = (1 - b_t_values)

    # a_t_hat := Product(a_i, i=1,...,t)
    # Note: to access a_t and a_t_hat use index t-1
    # A more efficient implementation of:
    # jnp.array([a_t_values[:t].prod() for t in range(1, T)])
    a_t_hat_values = jnp.cumprod(a_t_values)

    return a_t_hat_values, a_t_values


@jax.jit
def get_t_embedding(t):
    """
    From ChatGPT
    Sinusoidal time embedding (like in transformer positional encodings).
    t is () and we return an embedding of shape (embed_dim).
    t is shape () (scalar) because vmap gives us each sample
    """
    embed_dim = 128
    # Convert t to float32 for numerical safety
    t_float = t.astype(jnp.float32)

    half_dim = embed_dim // 2
    freqs = jnp.exp(
        - jnp.log(10000) * jnp.arange(0, half_dim, 1) / float(half_dim)
    )
    # shape (batch, half_dim)
    args = t_float * freqs
    embedding_sin = jnp.sin(args)
    embedding_cos = jnp.cos(args)
    embedding = jnp.concatenate([embedding_sin, embedding_cos], axis=0)
    return embedding


@functools.partial(jax.jit, static_argnames=['num_h_layers'])
@functools.partial(jax.vmap, in_axes=(None, None, 0, 0))
def ddpm_ffn_model_fn(params: dict, num_h_layers: int, x_noisy: jnp.array, t: jnp.array):
    """
    The epsilon_theta transformation of the noisy image and t,
    but simplified: using FFN instead of U-net transformer
    """
    # assert x_noisy.shape[0] + 1 == params['layer1']['W'].shape[0]
    t_emb = get_t_embedding(t)
    input_array = jnp.concatenate(
        arrays=(x_noisy, t_emb), axis=-1) #jnp.expand_dims(t, axis=-1)), axis=-1)

    # Forward pass through the network
    hid_state = input_array
    for layer_id in range(num_h_layers):
        layer_name = 'layer{}'.format(layer_id)
        hid_state = jnp.matmul(hid_state, params[layer_name]['W']) + params[layer_name]['b']
        hid_state = nn.gelu(hid_state)
    out = jnp.matmul(hid_state, params['projection']['W']) + params['projection']['b']

    return out


@functools.partial(jax.jit, static_argnames=['model_fn', 'num_h_layers'])
def sample_step_ddpm(
        params: jnp.array, num_h_layers: int, model_fn: typing.Callable,
        x_t: jnp.array, t: jnp.array, z: jnp.array,
        a_t_values: jnp.array, a_t_hat_values: jnp.array
):
    """
    Do one sample step (out of T) of image sampling (Algorithm 2). (https://arxiv.org/abs/2006.11239)
    :input: x_t: the value of x_t
    :return: x_t: the value of x_(t-1)
    """
    # Intermediate variables, for cleanliness
    a_t_coefficient = (1 - a_t_values[t - 1]) / jnp.sqrt(1 - a_t_hat_values[t - 1])
    eps_theta = model_fn(params, num_h_layers, x_t, jnp.array([t]))
    x_t_minus_eps_t = x_t - a_t_coefficient * eps_theta

    # (Sigma_t)^2 can be either b_t or (1-a_(t-1)^hat)/(1-a_t^hat)*b_t
    # (https://arxiv.org/abs/2006.11239)
    # Note: beta_t = 1 - alpha_t
    sigma_t = jnp.sqrt(1 - a_t_values[t - 1])

    # compute x_(t-1) by overwriting x_t
    x_t = (1 / jnp.sqrt(a_t_values[t - 1])) * x_t_minus_eps_t + sigma_t * z

    return x_t


def sample_ddpm_image(
        params: dict, num_h_layers: int, model_fn: typing.Callable, image_array_shape: tuple,
        T: int, a_t_values: jnp.array, a_t_hat_values: jnp.array, seed: int):
    """
    Implementing sampling in DDPM for image reconstruction from noise,
    as described in Algorithm 2 (https://arxiv.org/abs/2006.11239)
    :returns x_0, the reconstructed image
    """
    # First, sample x_T ~ N(0, I), in the image_array_shape
    x_t = jrand.normal(key=jrand.PRNGKey(seed=seed + 22435), shape=image_array_shape)

    # Compute x_(T-1),...,x_0 iteratively:
    for t in range(T, 0, -1):
        if t > 1:
            z_seed = t * (seed + 23509)
            z = jrand.normal(key=jrand.PRNGKey(seed=z_seed), shape=image_array_shape)
        else:
            z = jnp.zeros(shape=image_array_shape)

        # Run an optimised sampling step
        # Avoid using jax.jit on the main "for" loop because jit compilation is too slow
        x_t = sample_step_ddpm(
            params, num_h_layers, model_fn, x_t, t, z,
            a_t_values=a_t_values, a_t_hat_values=a_t_hat_values)

    # return what is essentially x_0, the reconstructed image
    return x_t


@functools.partial(jax.jit, static_argnames=['model_fn', 'num_h_layers'])
def compute_ddpm_loss(
        params: dict, num_h_layers: int, x: jnp.array, eps: jnp.array,
        t: jnp.array, a_t_hat_values: jnp.array, model_fn: typing.Callable):
    """
    Implementing the objective function of DDPM,
    as described in Algorithm 1 (https://arxiv.org/abs/2006.11239)
    """
    # First, forward x through the network
    # This corresponds to finding epsilon_theta in Algorthm 1
    a_t_hat = jnp.expand_dims(a_t_hat_values[t - 1], axis=1)
    x_noisy = jnp.sqrt(a_t_hat) * x + jnp.sqrt(1 - a_t_hat) * eps
    eps_theta = model_fn(params, num_h_layers, x_noisy, t)

    # Compute the MSE loss
    l = ((eps - eps_theta) ** 2).mean()

    return l


@functools.partial(jax.jit, static_argnames=['model_fn', 'num_h_layers', 'optim', 'T'])
def grad_and_update_ddpm(
        model_fn: typing.Callable, params: dict, num_h_layers: int, optim, opt_state,
        x: jnp.array, T: int, a_t_hat_values: jnp.array, seed: int):
    """
    This function implements Algorithm 1 from the DDPM work
    https://arxiv.org/abs/2006.11239
    """
    # We skip sampling x because we do it during data shuffling
    # Now, sample t and epsilon as t ~ Uniform({1, ..., T})
    # Get a uniform distribution from a categorical one with equal prob-s:
    uniform_logits = jnp.ones((T, x.shape[0]))
    t = jrand.categorical(
        key=jrand.PRNGKey(seed=seed), logits=uniform_logits, axis=0
    ) + 1

    # Then sample epsilon ~ N(0, I), of the same shape as x
    # Note: the sample is independent across both the batch and image dimensions
    eps = jrand.normal(key=jrand.PRNGKey(seed=seed+325), shape=x.shape)

    # Compute the gradients
    loss_value, grads = value_and_grad(compute_ddpm_loss)(
        params, num_h_layers, x, eps, t, a_t_hat_values, model_fn=model_fn)

    # Update the optimiser and the parameters
    updates, opt_state = optim.update(updates=grads, state=opt_state, params=params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, eps, t, loss_value


def run_ddpm_epoch(
        model_fn: typing.Callable, params: dict, num_h_layers: int, T: int,
        a_t_hat_values: jnp.array, optim, opt_state, x_train_data: jnp.array, x_dev_data: jnp.array,
        eval_interval: int=10, seed: int=32598):
    """
    T: (int), the number of diffusion steps
    """
    total_loss = 0
    num_loss_values = 0
    for batch_id, x in tqdm(enumerate(x_train_data), total=x_train_data.shape[0]):

        # Use a different seed for each iteration
        iter_seed = seed + 325 * (batch_id + 1)

        # Use the data to compute the gradients and update the optimiser and the parameters
        # This is done in a separate function to enable jax.jit optimisation with compiling
        params, opt_state, eps, t, loss_value = grad_and_update_ddpm(
            model_fn=model_fn, params=params, num_h_layers=num_h_layers,
            a_t_hat_values=a_t_hat_values, optim=optim, opt_state=opt_state,
            x=jnp.array(x), T=T, seed=iter_seed)

        # Record the training loss
        num_loss_values += 1
        total_loss += loss_value

        if (batch_id + 1) % eval_interval == 0:
            print('Training loss:', total_loss / (batch_id+1))
            # Dev evaluation
            # ToDo: implement this
            pass

    mean_train_loss = total_loss / num_loss_values

    return params, optim, opt_state, mean_train_loss
