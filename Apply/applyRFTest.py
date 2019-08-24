
import matplotlib.pyplot as plt
import numpy
from math import sin, cos
from sklearn.preprocessing import PolynomialFeatures
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


x_array = numpy.array(Aop)
y_arrar = numpy.array(Ber)

polynomial= PolynomialFeatures(degree=4)
polynomial.fit(x_array,y_arrar)





def get(AopMin, i):
    return AopMin+stepSize*i


def get31Array(x):
    X = []
    X.append(1)
    for i in range(15):
        X.append(sin(i/2*x))
        X.append(cos(i/2*x))
    return X


class_y = [get31Array(x) for x in Aop]
x_array = numpy.transpose(numpy.array(Ber))
y_arrar = numpy.array(class_y)

print(x_array.shape)
print(y_arrar.shape)
x = numpy.linalg.lstsq(y_arrar, x_array, rcond=None)[0]

res = numpy.transpose(x)
x_array = numpy.transpose(numpy.array(Ber))

Count = 100
stepSize = round((AopMax-AopMin)/100.0, 2)
new_Aop = [get(AopMin, i) for i in range(Count)]
new_xx_array=numpy.array(new_Aop)
new_x_array =numpy.array([get31Array(x) for x in new_Aop])
print(new_x_array)
print(new_x_array.shape)
print(res.shape)
dd=numpy.matmul(new_x_array,res)
print(dd)

plt.plot(new_xx_array,dd)
plt.show()
