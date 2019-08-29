from numpy import *
import numpy as np

i2 = eye(2)
print(i2)

savetxt("eye.txt", i2)

MyArray = np.array([[0.00028179, 0.00019766], [0.00019766, 0.00030123]])
print(MyArray)


# standard deviation
print(MyArray.std())
# Covariance 协方差

# Covariance
print(MyArray.diagonal())


# Trace
print(MyArray.trace())

# Correlation from covariance


# Determine the signs
change = np.array([1.92, -1.08, -1.26, 0.63, -1.54, -0.28, 0.25, -0.6, 2.15, 0.69, -1.33, 
 1.16, 1.59, -0.26, -1.29, -0.13, 
-2.12, -3.91, 1.28, -0.57, -2.07, 2.07, 2.5, 1.18, -0.88, 1.31, 1.24, -.59])

signs = np.sign(change)
pieces = np.piecewise(change, [change < 0, change > 0], [-1, 1])
print(signs)