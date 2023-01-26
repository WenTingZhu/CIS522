import numpy as np


class LinearRegression:

    # w: np.ndarray
    # b: float

    def __init__(self):
        self.w = []
        self.b = 0

    def fit(self, X, y):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        if np.linalg.det(X.T @ X):
            weights = np.linalg.inv(X.T @ X) @ X.T @ y
            self.w = weights[:-1]
            self.b = weights[-1]
            # print(self.w.shape, self.b.shape)
            # print(self.w, self.b)
            # print(X.shape)

    def predict(self, X):
        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    # Private method to calculate MSE
    def _mean_squared_error(y_true, y_predicted):
        return np.sum((y_true - y_predicted)**2) / len(y_true)

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        n = float(len(x))
        for i in range(epochs):
            y_predicted = self.w @ X + self.b
            # loss = (y - y_hat) ** 2
            # gradient_w = 2 * X * np.abs(y - y_hat)
            # gradient_b = 2 * np.abs(y - y_hat)
            # Calculating the gradients
            gradient_w = -(2/n) * sum(X * (y - y_predicted))
            gradient_b = -(2/n) * sum(y - y_predicted)

            w -= gradient_w * lr
            b -= gradient_b * lr
        return (w, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return self.w @ X + self.b
