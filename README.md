# Image Models and Diffusion in JAX

This is a coding exercise in implementing image models and diffusion in JAX.

**Note:** This is a work in development; for educational purposes only.

**Note:** I am new to JAX, 
diffusion, and image models.

Using JAX from: [https://github.com/jax-ml/jax](https://github.com/jax-ml/jax).


## Setup and Installation

1. Download and unzip the MNIST dataset from https://yann.lecun.com/exdb/mnist/, or from a mirror such as:
https://github.com/mkolod/MNIST.

2. Run in terminal:

```
conda create -n jax optax python anaconda
```

3. For NVIDIA GPUs, run in terminal:

```
pip3 install -U "jax[cuda12]"
```

For other hardware see the [JAX documentation](https://github.com/jax-ml/jax).
