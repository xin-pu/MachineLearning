import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


x = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2],
              [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
y = np.array([0, 0, 0, 1,  1, 1, 2, 2, 2])


def plot_classifier(classifier, x, y):

    x_min, x_max = min(x[:, 0])-1.0, max(x[:, 0])+1.0
    y_min, y_max = min(x[:, 1])-1.0, max(x[:, 1])+1.0

    stepSize = 0.01

    x_steps, y_steps = np.meshgrid(
        np.arange(x_min, x_max, stepSize), np.arange(y_min, y_max, stepSize))
    mesh_output = classifier.predict(np.c_[x_steps.ravel(), y_steps.ravel()])
    mesh_output = mesh_output.reshape(x_steps.shape)
    print(x_steps)
    print(y_steps)
    plt.figure()
    plt.pcolormesh(x_steps, y_steps, mesh_output, cmap=plt.cm.gray)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=80, edgecolors='black',
                linewidths=1, cmap=plt.cm.Paired)
    plt.xlim(x_steps.min(), x_steps.max())
    plt.ylim(y_steps.min(), y_steps.max())
    plt.show()


classifier = linear_model.LogisticRegression(solver='liblinear', C=100)
classifier.fit(x, y)
plot_classifier(classifier, x, y)
