import numpy as np
import abc


class Loss(abc.ABC):
    @abc.abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the loss between the predicted and true values.

        Parameters:
            y_pred (`np.ndarray`):
                The predicted values, typically output from a model. The shape
                of `y_pred` is (`batch_size`, `input_dim`).
            y_true (`np.ndarray`):
                The true/target values corresponding to the predictions. Note
                that the shapes of `y_pred` and `y_true` might not match, but
                they will have the same number of samples, i.e. `batch_size`.

        Returns:
            loss (`np.ndarray`):
                The loss value for each sample in the input. The shape of the
                output is (`batch_size`,).
        """
        return NotImplemented

    @abc.abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. the predicted values.

        Parameters:
            grad (`np.ndarray`):
                The gradient of the final loss with respect to the output of
                `self.forward`. This value typically reflects a scaling factor
                caused by averaging the loss, and can also be used to reverse
                the gradient direction to achieve gradient ascent.

        Returns:
            grad (`np.ndarray`):
                The gradient of the loss with respect to the `y_pred`.
        """
        return NotImplemented


class CrossEntropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the cross-entropy loss between the predicted probabilities and
        the true labels.

        Parameters:
            y_pred (`np.ndarray`):
                A 2D array of predicted probabilities, where the first axis
                corresponds to the number of samples and the second axis
                corresponds to the number of classes.
            y_true (`np.ndarray`):
                A 1D array of true labels, where each label is an integer in
                the range [0, num_classes).
        """
        #防止log0
        safe = 1e-6
        y_pred = np.clip(y_pred, safe, 1 - safe)
        batch_size = y_pred.shape[0]
        #保存y_pred, y_true給backward
        self.cache = (y_pred,y_true)
        #損失
        loss = - np.log(y_pred[np.arange(batch_size), y_true])

        return np.sum(loss)


    def backward(self, grad: np.ndarray) -> np.ndarray:
        #-1/ypred
        y_pred,y_true = self.cache
        batch_size = y_true.shape[0]

        grad_input = y_pred.copy()
        grad_input[np.arange(batch_size), y_true] -= 1
        grad_input /= batch_size
        return grad_input # * grad[:, np.newaxis]
