"""
This is an implementation of diffusion models in JAX
"""

import jax.numpy as jnp
import jax.nn as nn  # activation fn-s

from jax import random, grad, jit, vmap


# First, let's implement a simple FFN for images, to learn the JAX basics
class FFN:
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
        assert x.shape[0] * x.shape[1] == self.in_size
        x = x.reshape(1, -1)  # flatten the image dimensions

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
            print(out)
            print(predictions)


    def run_epoch(self, x_data, y_data):
        for batch_id, (x, y) in enumerate(zip(x_data, y_data)):
            out = self.forward(x)
            print('Output:', out)
            loss = self.cross_entropy_loss(out, y)
            print('Loss:', loss)

            if batch_id % 10 == 0:
                # Training evaluation
                # ToDo: replace this with dev evaluation
                self.evaluate(x_data, y_data)
            #loss.backward()


"""
# https://github.com/jax-ml/jax/blob/main/README.md
def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)  # inputs to the next layer
  return outputs                # no activation on last layer


def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2)


grad_loss = jit(grad(loss))  # compiled gradient evaluation function
perex_grads = jit(vmap(grad_loss, in_axes=(None, 0, 0)))  # fast per-example grads


# Batched application of "apply_matrix" (https://jax.readthedocs.io/en/latest/quickstart.html)
@jit
def vmap_batched_apply_matrix(batched_x):
  return vmap(apply_matrix)(batched_x)


np.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                           vmap_batched_apply_matrix(batched_x), atol=1E-4, rtol=1E-4)
print('Auto-vectorized with vmap')
%timeit vmap_batched_apply_matrix(batched_x).block_until_ready()
"""
