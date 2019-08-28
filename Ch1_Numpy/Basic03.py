import numpy as np
import random
from math import sin, pi
import matplotlib.pyplot as plt

v = np.array([[2], [1], [3]])
print(v)

print(np.transpose(np.array([[2, 1, 3]])))


A = np.array([[2, 1, -2], [3, 0, 1], [1, 1, -1]])
b = np.transpose(np.array([[-3, 5, -2]]))


x = np.linalg.solve(A, b)
print(x)


def sin_ext(x):
    pix = pi*x
    return sin(pix)/pix+0.1*x+0.05*random.randint(1, 100)


d = [sin_ext(i) for i in range(1,100)]
print(d)
plt.plot(d)
plt.show()