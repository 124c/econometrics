import pandas as pd
import numpy as np
import utils.utils as utils


def get_betas(X, Y):
    """
    :param X: numpy matrix or pd.DataFrame
    :param Y: numpy array, pd.DataFrame or pd.Series
    :return: np.array
    """
    X_vals = utils.format_X(X)
    XTX = np.dot(X_vals.T, X_vals)
    XTX_min1 = np.linalg.inv(XTX)

    Y_vals = utils.format_Y(Y)
    XTY = np.dot(X_vals.T, Y_vals)

    B_hat = np.dot(XTX_min1, XTY)
    return B_hat


def get_Y_hat(X, Y):
    """
    :param X:
    :return:
    """
    B_hat = get_betas(X, Y)
    Y_hat = np.dot(X, B_hat)
    return Y_hat


