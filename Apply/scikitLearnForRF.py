
import matplotlib.pyplot as plt
import numpy
from math import sin, cos
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
filename = "E:\Document Code\Code Pensonal\MachineLearning\Apply\RFTestData.csv"

Aop = []
Ber = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split('\t')]
        Aop.append(data[0])
        Ber.append(data[1])

AopMin = min(Aop)
AopMax = max(Aop)


x_array = numpy.array(Aop).reshape(-1,1)
y_array = numpy.array(Ber).reshape(-1,1)
print(x_array.shape)
print(y_array.shape)
polynomial= PolynomialFeatures(degree=15)
tt=polynomial.fit_transform(x_array)

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(tt,y_array)

y_pred=linear_regressor.predict(tt)

y_pred_point=linear_regressor.predict(polynomial.fit_transform(numpy.array([-8]).reshape(-1,1)))
plt.plot(x_array,y_array,color='black')
plt.plot(x_array,y_pred,color='red')
plt.show()
print(y_pred_point)