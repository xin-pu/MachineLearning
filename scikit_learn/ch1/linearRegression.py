import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X=[]
Y=[]
with open("e:/Document Code/Code Pensonal/MachineLearningGitHub/scikit_learn/ch1/data.csv",'r') as f:
    for line in f.readlines():
        xt,yt=[float(i) for i in line.split(',')]
        X.append(xt)
        Y.append(yt)

num_traning=int(0.8*len(X))
num_test=len(X)-num_traning

X_train=np.array(X[:num_traning]).reshape((num_traning,1))
Y_train=np.array(Y[:num_traning])

X_test=np.array(X[num_traning:]).reshape((num_test,1))
Y_test=np.array(Y[num_traning:])

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train,Y_train)

y_train_pred=linear_regressor.predict(X_train)
plt.figure()

plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,y_train_pred,color="black",linewidth=2)
plt.show()

y_test_pred=linear_regressor.predict(X_test)
plt.scatter(X_test,Y_test,color="red")
plt.plot(X_test,y_test_pred,color="black",linewidth=2)
plt.show()


#Step 2 Metric
import sklearn.metrics as sm

print(sm.mean_absolute_error(Y_test,y_test_pred))
print(sm.mean_squared_error(Y_test,y_test_pred))
print(sm.median_absolute_error(Y_test,y_test_pred))
print(sm.explained_variance_score(Y_test,y_test_pred))
print(sm.r2_score(Y_test,y_test_pred))



