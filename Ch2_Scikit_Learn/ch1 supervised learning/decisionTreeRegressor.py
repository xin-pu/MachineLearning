from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as sm
import numpy as np
import matplotlib.pyplot as plt

housing_data = datasets.load_boston()

X, Y = shuffle(housing_data.data, housing_data.target, random_state=7)

print(X)
num_traning = int(0.8*len(X))
num_test = len(X)-num_traning

X_train = np.array(X[:num_traning])
Y_train = np.array(Y[:num_traning])

X_test = np.array(X[num_traning:])
Y_test = np.array(Y[num_traning:])

dr_regressor = DecisionTreeRegressor(max_depth=4)
dr_regressor.fit(X_train, Y_train)

y_test_pred = dr_regressor.predict(X_test)
# print(sm.mean_absolute_error(Y_test, y_test_pred))
# print(sm.mean_squared_error(Y_test, y_test_pred))
# print(sm.median_absolute_error(Y_test, y_test_pred))
# print(sm.explained_variance_score(Y_test, y_test_pred))
# print(sm.r2_score(Y_test, y_test_pred))


def plot_feature_importances(feature_importances, feature_names):
    feature_importances = 100*feature_importances/max(feature_importances)
    index_sorted = np.flipud(np.argsort(feature_importances))
    pos = np.arange(index_sorted.shape[0])+0.5
    plt.bar(pos, feature_importances[index_sorted], color='Green')
    plt.xticks(pos, feature_names[index_sorted])
    plt.show()


plot_feature_importances(
    dr_regressor.feature_importances_, housing_data.feature_names)
