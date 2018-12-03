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
x_data = {'a':  [1., 1., 1., 1., 1.],
          'x1': [0., 1., 0., 1., 1.],
          'x2': [0., 0., 0., 0., 1.]}
X = pd.DataFrame(data=x_data).as_matrix()  # Method .as_matrix will be removed in a future version. Use .values

XTX = np.dot(X.T, X)
XTX_min1 = np.linalg.inv(XTX).round(decimals=5)
Y_hat = ols.get_Y_hat(X, Y)
# a) show the number of observations
n = len(Y)
# b) show the number of regressors in a model (with alpha)
k = X.shape[1]
# c) Calculate TSS
TSS = metrics.TSS(Y, np.mean(Y))
# d) Calculate Y_hat with OLS
b_hat = ols.get_betas(X, Y)
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
# RSS/B = Ch1 <= σ^2 <= RSS/A = Ch2
Ch1 = RSS/B
Ch2 = RSS/A
sigma_squared_confint = [Ch1, Ch2]
# i) Calculate 80% confidence int for σ
sigma_confint = [np.sqrt(Ch1), np.sqrt(Ch2)]
# j) Calculate unbiased estimate of covariance matrix
# v_hat(β_hat) = σ^2(XTX^-1)
v_hat = sigma2 * XTX_min1
# k) Calculate 90% confint for β1
# st = (β1_hat - β1)/sqrt(D_hat(β1_hat)) ~ t(n-k)
# P({A <= st <= B}) = 0.9 (A and B are critical values for distribution)
A_t = scipy.stats.t.ppf(q=0.05, df=sigma2).round(decimals=5)
B_t = scipy.stats.t.ppf(q=0.95, df=sigma2).round(decimals=5)
# data for plotting
t_plot_data = scipy.stats.t.rvs(df=sigma2, size=1000)
# the histogram of the data
plt.hist(t_plot_data, 50, density=True, facecolor='orange', alpha=0.75)
plt.axvline(x=A)
plt.axvline(x=B)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Histogram of t {}'.format(sigma2))
plt.show()
# now the interval itself: A <= st <= B means that
# A*sqrt(D_hat(β_hat1)) = Tcr1 <= β_hat1-β1 <= Tcr2 = B*sqrt(D_hat(β_hat1))
Tcr1 = b_hat[1] - B_t*np.sqrt(v_hat[1][1])
Tcr2 = b_hat[1] - A_t*np.sqrt(v_hat[1][1])
beta1_confint = [Tcr1, Tcr2]
# l = m = n = o = p
# p) Calculate 95% confidence interval for α+2β1-3β2 = l(β)
# P({T1 <= l(β) <= T2}) = 0.95,
# st_p = (l(β_hat) - l(β))/sqrt(D_hat(l(β_hat))) ~ t(n-k)
A_tp = scipy.stats.t.ppf(q=0.025, df=sigma2).round(decimals=5)
B_tp = scipy.stats.t.ppf(q=0.975, df=sigma2).round(decimals=5)
# TODO: p-s
# now the interval itself: A <= st_p <= B means that
# A_tp*sqrt(D_hat(β_hat)) = Tcr_p1 <= β_hat-β <= Tcr_p2 = B_tp*sqrt(D_hat(β_hat))
# Tcr_p1 = b_hat[1] - B_t*np.sqrt(v_hat[1][1])
# Tcr_p2 = b_hat[1] - A_t*np.sqrt(v_hat[1][1])
# p_confint = [A_tp, B_tp]
