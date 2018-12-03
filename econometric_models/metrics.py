import pandas as pd
import numpy as np


def TSS(Y, Y_mean):
    """
    :param Y: np.array
    :param Y_mean: float
    :return: float
    """
    return sum((Y - Y_mean) ** 2)


def RSS(Y, Y_hat):
    """
    Basically what we calculate is sum(e**2) - standard deviation of errors
    :param Y: np.array
    :param Y_hat: np.array
    :return: float
    """
    return sum((Y - Y_hat) ** 2)


def ESS(Y_mean, Y_hat):
    """
    :param Y_mean: float 
    :param Y_hat: np.array
    :return: float
    """
    return sum((Y_hat-Y_mean) ** 2)


def R2(Y, Y_hat):
    r2 = ESS(np.mean(Y), Y_hat)/TSS(Y,np.mean(Y))
    return r2


def sigma_squared(rss, n, k):
    sigma2_val = rss/(n-k)
    return sigma2_val