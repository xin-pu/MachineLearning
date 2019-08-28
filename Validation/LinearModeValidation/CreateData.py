
import numpy as np
import numpy.matlib as matlib
from numpy import *
import matplotlib.pyplot as plt
import random


Num = 100

x_data = np.linspace(-3, 3, Num)
x_datawithPi = x_data*pi
s = matlib.randn(Num)

print(x_data.shape)
newb = s.flatten()
s = np.ravel(newb)

Y_data = np.sin(x_datawithPi)/x_datawithPi+0.1*x_data+0.05*s
print(Y_data)


plt.scatter(x_data, Y_data, color="black", linewidth=2)
plt.xlim(-3, 3)
plt.ylim(-2, 2)
plt.show()

data = vstack((x_data, Y_data))
savetxt("Validation\LinearModeValidation\eye.txt", data.transpose())
