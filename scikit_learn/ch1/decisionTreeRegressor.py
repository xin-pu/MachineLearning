from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
import sklearn.metrics as sm
import numpy as np

housing_data = datasets.load_boston()

X, Y = shuffle(housing_data.data, housing_data.target, random_state=7)


num_traning=int(0.8*len(X))
num_test=len(X)-num_traning

X_train=np.array(X[:num_traning])
Y_train=np.array(Y[:num_traning])

X_test=np.array(X[num_traning:])
Y_test=np.array(Y[num_traning:])

dr_regressor=DecisionTreeRegressor(max_depth=4)
dr_regressor.fit(X_train,Y_train)

y_test_pred=dr_regressor.predict(X_test)
print(sm.mean_absolute_error(Y_test,y_test_pred))
print(sm.mean_squared_error(Y_test,y_test_pred))
print(sm.median_absolute_error(Y_test,y_test_pred))
print(sm.explained_variance_score(Y_test,y_test_pred))
print(sm.r2_score(Y_test,y_test_pred))


