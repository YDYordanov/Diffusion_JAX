"""
This is an implementation of diffusion models in JAX
"""

import jax.numpy as jnp
import jax.nn as nn  # activation fn-s

from jax import random, grad, jit, vmap


class FFN:
    """
    First, let's implement a simple FFN for images, to learn the JAX basics
    """
    def __init__(self, in_size: int, h_size: int, out_size: int):
        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size

        # First layer
        self.W = random.normal(random.PRNGKey(12), (self.in_size, self.h_size))
        self.b = random.normal(random.PRNGKey(1322), (self.h_size))

        # Projection layer
        self.proj_W = random.normal(random.PRNGKey(23), (self.h_size, 1))
        self.proj_b = random.normal(random.PRNGKey(125), (self.out_size))


    def forward(self, x):
        assert x.shape[1] * x.shape[2] == self.in_size
        x = x.reshape(x.shape[0], -1)  # flatten the image dimensions

        hid_state = jnp.matmul(x, self.W) + self.b
        hid_state = nn.gelu(hid_state)
        out = jnp.matmul(hid_state, self.proj_W) + self.proj_b
        return out


    def cross_entropy_loss(self, out, y):
        # First, compute softmax of the outputs to obtain probabilities
        out_probs = nn.softmax(out, axis=-1)

        # 1-hot encode the targets y
        y_vector = nn.one_hot(y, num_classes=self.out_size)

        # Compute log-likelihood loss w.r.t. y_vector
        l = - (jnp.log(out_probs) * y_vector).sum(axis=-1)

        return l


    def evaluate(self, x_test_data, y_test_data):
        for x, y in zip(x_test_data, y_test_data):
            out = self.forward(x)
            # Model predictions
            predictions = out.argmax(axis=-1)
            #print(out)
            #print(predictions)


    def run_epoch(self, x_train_data: jnp.array, y_train_data: jnp.array,
                  x_dev_data: jnp.array, y_dev_data: jnp.array):
        for batch_id, (x, y) in enumerate(zip(x_train_data, y_train_data)):
            out = self.forward(x)
            #print('Output:', out)
            loss = self.cross_entropy_loss(out, y)
            #print('Loss:', loss)

            if batch_id % 10 == 0:
                # Dev evaluation
                self.evaluate(x_dev_data, y_dev_data)
            #loss.backward()
