import numpy as np


class LinearRegression:
    """
    A class for Linear Regression.

    ...

    Attributes
    ----------
    w: np.ndarray
        weights
    b: float
        bias

    Methods
    -------
    fit(self, X: np.ndarray, y: np.ndarray):
        Fit the model with the data.

    predict(self, X: np.ndarray):
        Predict the output for the given input.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        """
        Constructs all the necessary attributes for the LinearRegression object.
        """

        self.w = []
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model with the data.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The true data.
            lr (float): The learning rate.
            epochs (int): The number of iterations to run the model

        Returns:
            np.ndarray: The predicted output.
        """

        X = np.hstack((X, np.ones((X.shape[0], 1))))
        if np.linalg.det(X.T @ X):
            weights = np.linalg.inv(X.T @ X) @ X.T @ y
            self.w = weights[:-1]
            self.b = weights[-1]

    def predict(self, X: np.ndarray):
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        return X @ self.w + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.

    """

    # Private method to calculate MSE
    def _mean_squared_error(y_true, y_predicted):
        return np.sum((y_true - y_predicted) ** 2) / len(y_true)

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the model with the data.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The true data.
            lr (float): The learning rate.
            epochs (int): The number of iterations to run the model

        Returns:
            None
        """

        n = float(len(X))
        for i in range(epochs):
            # Predict y
            y_predicted = X @ self.w + self.b

            # Calculating the gradients
            gradient_w = -(2 / n) * sum(X * (y - y_predicted))
            gradient_b = -(2 / n) * sum(y - y_predicted)

            self.w -= gradient_w * lr
            self.b -= gradient_b * lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """

        return X @ self.w + self.b
