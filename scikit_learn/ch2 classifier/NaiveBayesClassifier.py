from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

filename = "E:\Document Code\Code Pensonal\MachineLearning\scikit_learn\ch2 classifier\data_muktivar.txt"


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


X = []
Y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        Y.append(data[-1])

data_input = np.array(X)
data_output = np.array(Y)

classifier = GaussianNB()
classifier.fit(data_input, data_output)

data_output_pred = classifier.predict(data_input)

accuracy = 100.0 * (data_output == data_output_pred).sum() / \
    data_input.shape[0]
print(accuracy)
# plot_classifier(classifier, data_input, data_output)


x_train, x_test, y_train, y_test = train_test_split(
    data_input, data_output, test_size=0.25, random_state=5)
classifier.fit(x_train, y_train)
y_test_pred = classifier.predict(x_test)

accuracy = 100.0 * (y_test_pred == y_test).sum() / data_input.shape[0]
print(accuracy)
plot_classifier(classifier, x_test, y_test_pred)
