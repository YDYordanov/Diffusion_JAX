# Image Models and Diffusion in JAX

This is a coding exercise in implementing image models and diffusion in JAX.

**Note:** This is an amateur work in development; for educational purposes only. Any code originating from an external 
source (e.g. ChatGPT) except for official documentation is clearly labelled as such.

Using JAX from: [https://github.com/jax-ml/jax](https://github.com/jax-ml/jax).

The diffusion model being implemented here is based on the DDPM model by: https://arxiv.org/abs/2006.11239.

**Note:** High GPU memory consumption is normal because JAX reserves a large percentage of the GPU memory 
at the first use of JAX. For more details, see: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html.

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
pip install -U "jax[cuda12]" "ray[tune]" optuna
```

For other hardware see the [JAX documentation](https://github.com/jax-ml/jax).

## Example Usage

```
python main.py -data=cifar10 -model=ffn -lr=1e-4 --num_epochs=10
```
