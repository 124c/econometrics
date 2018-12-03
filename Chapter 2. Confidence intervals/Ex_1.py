# import base packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import specializes packages
import scipy.stats
# import self-written packages
import econometric_models.metrics as metrics
import econometric_models.OLS as ols

# we are given a regression model:
# y = a + β1xi1 + β2xi2 + ei, i = 1,...,n
# e ~ N(0, sigma_squaredI)
# in a matrix form:
# Y = Xβ + e

Y = [1, 2, 3, 4, 5]
x_data = {'a':  [1, 1, 1, 1, 1],
          'x1': [0, 1, 0, 1, 1],
          'x2': [0, 0, 0, 0, 1]}
X = pd.DataFrame(data=x_data).as_matrix()  # Method .as_matrix will be removed in a future version. Use .values

XTX = np.dot(X.T, X)
XTX_min1 = np.linalg.inv(XTX)

# a) show the number of observations
n = len(Y)
# b) show the number of regressors in a model (with alpha)
k = X.shape[1]
# c) Calculate TSS
TSS = metrics.TSS(Y, np.mean(Y))
# d) Calculate Y_hat with OLS
Y_hat = ols.get_Y_hat(X, Y)
# e) Calculate RSS
RSS = metrics.RSS(Y, Y_hat)
# f) Calculate R2
R2 = metrics.R2(Y, Y_hat)
# g) Calculate unbiased σ^2 for the model
sigma2 = metrics.sigma_squared(RSS, n, k)
# h) Calculate 80% confidence int for σ^2
# assuming that RSS/σ^2 ~ chisq(n-k), we shall find such A and B coefs that
# P({A <= RSS/σ^2 <= B}) = 0.8 (A and B are critical values for distribution)
A = scipy.stats.chi2.ppf(q=0.1, df=sigma2)
B = scipy.stats.chi2.ppf(q=0.9, df=sigma2)
# data for plotting
chi_plot_data = scipy.stats.chi2.rvs(df=sigma2, size=1000)
# the histogram of the data
plt.hist(chi_plot_data, 100, density=True, facecolor='orange', alpha=0.75)
plt.axvline(x=A)
plt.axvline(x=B)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Histogram of chisq {}'.format(sigma2))
plt.show()
# now the interval itself: A <= RSS/(n-k) <= B means that
# RSS/B = T1 <= σ^2 <= RSS/A = T2
T1 = RSS/B
T2 = RSS/A
sigma_squared_confint = [T1, T2]
# i) Calculate 80% confidence int for σ
sigma_confint = [np.sqrt(T1), np.sqrt(T2)]
# j) Calculate 90% confint for β1
