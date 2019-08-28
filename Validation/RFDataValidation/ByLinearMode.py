
import matplotlib.pyplot as plt
import numpy
from math import sin, cos
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
filename = "Validation\RFDataValidation\RFTestData.csv"

mode = ["OrdinaryLeastSquares", "RidgeRegression"]

Aop = []
Ber = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split('\t')]
        Aop.append(data[0])
        Ber.append(data[1])


AopMin, AopMax = min(Aop), max(Aop)
length = len(Aop)

x_array = numpy.array(Aop).reshape((length, 1))
y_arrar = numpy.array(Ber).reshape((length, 1))

polynomial = PolynomialFeatures(degree=12)
x_array_polynomial = polynomial.fit_transform(x_array)

mode = 1
if mode == 1:
    linear = LinearRegression()
elif mode == 2:
    linear = Ridge(alpha=.5)
elif mode == 3:
    linear = Lasso(alpha=.01)
elif mode == 4:
    linear = SVR()
elif mode == 5:
    linear = BayesianRidge()

linear.fit(x_array_polynomial, y_arrar)
y_pred = linear.predict(x_array_polynomial)

plt.scatter(x_array, y_pred, color="red")
plt.plot(x_array, y_arrar, color="black", linewidth=2)
plt.xlim(x_array.min(), x_array.max())
plt.ylim(y_arrar.min()-0.0001, y_arrar.max()+0.0002)
plt.show()

# The result is: use  degree >= 10 by LinearRegression and Ridge can better fit the result. 
