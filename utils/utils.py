import pandas as pd
import numpy as np


def format_X(X):
    if (isinstance(X, pd.DataFrame)) or(isinstance(X, pd.Series)):
        X_vals = X.as_matrix()
    else:
        X_vals = X
    return X_vals


def format_Y(Y):
    if (isinstance(Y, pd.DataFrame)) or(isinstance(Y, pd.Series)):
        Y_vals = Y.values()
    else:
        Y_vals = Y
    return Y_vals
