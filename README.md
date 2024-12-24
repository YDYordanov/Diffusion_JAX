# Image Models and Diffusion in JAX

This is a coding exercise in implementing image models and diffusion in JAX.

**Note:** This is an amateur work in development; for educational purposes only.

**Note:** high GPU memory consumption may be due to JAX's default memory allocation behaviour. For more details, see: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html.

Using JAX from: [https://github.com/jax-ml/jax](https://github.com/jax-ml/jax).


## Setup and Installation

1. Download and extract the data files of the datasets you need:

    MNIST can be found at: https://yann.lecun.com/exdb/mnist/, or from a mirror such as:
https://github.com/mkolod/MNIST.

    CIFAR-10 can be found at: https://www.cs.toronto.edu/~kriz/cifar.html.


2. Run in terminal:

```
conda create -n jax_env optax python anaconda
conda activate jax_env
```

3. For NVIDIA GPUs, run in terminal:

```
pip3 install -U "jax[cuda12]"
```

For other hardware see the [JAX documentation](https://github.com/jax-ml/jax).

## Example Usage

```
python main.py -data=cifar10 -lr=1e-4 --num_epochs=10
```
