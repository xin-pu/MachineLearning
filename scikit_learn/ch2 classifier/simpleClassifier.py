import numpy as np
import matplotlib.pyplot as plt


x = np.array([[3, 1], [2, 5], [1, 8], [6, 4], [5, 2], [3, 5], [4, 7], [4, -1]])
y = [0, 1, 1, 0, 0, 1, 1, 0]

class_0 = np.array([x[i] for i in range(len(x)) if y[i] == 0])
class_1 = np.array([x[i] for i in range(len(x)) if y[i] == 1])

plt.scatter(class_0[:, 0], class_0[:, 1], color="Green", marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], color="Red", marker='x')

x_l = range(10)
y_l = x_l
plt.plot(x_l, y_l, color='Black')

plt.show()
