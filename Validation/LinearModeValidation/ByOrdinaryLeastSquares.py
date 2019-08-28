
import matplotlib.pyplot as plt
import numpy
from math import sin, cos
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn import datasets


data = numpy.loadtxt('Validation\LinearModeValidation\eye.txt')
print(data)
X = data[:, 0].reshape(-1,1)
Y = data[:, 1].reshape(-1,1)
x_pred = numpy.linspace(-3, 3, 500).reshape(-1,1)


mode = 2
if mode == 1:
    linear = LinearRegression()
elif mode == 2:
    linear = Ridge(alpha=.9)
elif mode == 3:
    linear = Lasso(alpha=.01)
elif mode == 4:
    linear = SVR()
elif mode == 5:
    linear = BayesianRidge()

polynomial = PolynomialFeatures(degree=15)
x_polynomial = polynomial.fit_transform(X)

x_pred_polymial=polynomial.fit_transform(x_pred)

linear = LinearRegression()
linear.fit(x_polynomial, Y)
y_pred=linear.predict(x_pred_polymial)

plt.scatter(X, Y, color="red")
plt.plot(x_pred, y_pred, color="black", linewidth=2)
plt.show()
