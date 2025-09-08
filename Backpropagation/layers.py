import abc
from typing import List

import numpy as np


class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propogate through the layer.

        Parameters:
            x (`np.ndarray`):
                input tensor with shape `(batch_size, input_dim)`.

        Returns:
            out (`np.ndarray`):
                output tensor of the layer with shape
                `(batch_size, output_dim)`.
        """
        return NotImplemented

    @abc.abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward propogate through the layer.

        Calculate the gradient of the loss with respect to both the input and
        the parameters of the layer. The gradients with respect to the
        parameters are summed over the batch.

        Parameters:
            grad (`np.ndarray`):
                gradient with respect to the output of the layer.

        Returns:
            grad (`np.ndarray`):
                gradient with respect to the input of the layer.
        """
        return NotImplemented

    def params(self) -> List[np.ndarray]:
        """Return the parameters of the parameters.

        The order of parameters should be the same as the order of gradients
        returned by `self.grads()`.
        """
        return []

    def grads(self) -> List[np.ndarray]:
        """Return the gradients of the parameters.

        The order of gradients should be the same as the order of parameters
        returned by `self.params()`.
        """
        return []


class Sequential(Layer):
    """A sequential model that stacks layers in a sequential manner.

    Parameters:
        layers (`List[Layer]`):
            list of layers to be added to the model.
    """
    def __init__(self, *layers: List[Layer]):
        self.layers = []
        self.append(*layers)

    def append(self, *layers: List[Layer]):
        """Add layers to the model"""
        self.layers.extend(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sequentially forward propogation through layers.

        Parameters:
            x (`np.ndarray`):
                input tensor.

        Returns:
            out (`np.ndarray`):
                output tensor of the last layer.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Sequentially backward propogation through layers.

        Parameters:
            grad (`np.ndarray`):
                gradient of loss function with respect to the output of forward
                propogation.

        Returns:
            grad (`np.ndarray`):
                gradient with respect to the input of forward propogation.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> List[np.ndarray]:
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.params())
        return parameters

    def grads(self) -> List[np.ndarray]:
        gradients = []
        for layer in self.layers:
            gradients.extend(layer.grads())
        return gradients


class MatMul(Layer):
    """Matrix multiplication layer.

    Parameters:
        W (`np.ndarray`):
            weight matrix with shape `(input_dim, output_dim)`.
    """
    def __init__(self, W: np.ndarray):
        self.W = W
        self.dW = np.zeros_like(W)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.dW += self.x.T @ grad
        dx = grad @ self.W.T
        return dx

    def params(self) -> List[np.ndarray]:
        return [self.W]

    def grads(self) -> List[np.ndarray]:
        return [self.dW]


class Bias(Layer):
    """Bias layer.

    Parameters:
        b (`np.ndarray`):
            bias vector with shape `(output_dim,)`.
    """
    def __init__(self, b: np.ndarray):
        self.b = b
        self.db = np.zeros_like(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.db += np.sum(grad, axis = 0)
        # 1 * grad
        return grad

    def params(self) -> List[np.ndarray]:
        return [self.b]

    def grads(self) -> List[np.ndarray]:
        return [self.db]


class ReLU(Layer):
    """Rectified Linear Unit."""

    def forward(self, x):
        self.cache = x
        return np.clip(x, 0, None)

    def backward(self, grad):
        x = self.cache
        return np.where(x > 0, grad, 0)


class Softmax(Layer):
    def forward(self, x):
        x -= np.max(x, axis = 1, keepdims = True)
        temp = np.exp(x)
        self.y = temp / np.sum(temp, axis = 1, keepdims = True)
        return self.y

    def backward(self, grad):
        # Softmax + CrossEntropy
        #n = self.y.shape[1]
        #batch_size = self.y.shape[0]  # Copy the softmax probabilities
        #grad_input[np.arange(grad.shape[0]), np.argmax(self.y, axis=1)] -= 1  # Simplified gradient
        #return grad_input * grad
        return self.y * (grad - np.sum(grad * self.y, axis=1, keepdims=True))
