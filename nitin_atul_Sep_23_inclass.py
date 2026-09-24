#Logistic Regression
import numpy as np
import pandas as pd
def hw_x(X, w):
    return 1/(1+(np.e)**(-(X@w)))
def J_w(X, y, w):
    return (-1/X.shape[0])*sum(((y*np.log(hw_x(X, w))) + ((1-y)*np.log(1-hw_x(X, w)))))
def grad_J_w(X, y, w, alpha):
    return (w - ((alpha/X.shape[0])*(np.transpose(hw_x(X, w) - y)@X)))

pandas_raw = pd.read_csv("iris_data.csv")

#Turn into Array for numpy stuff
iris_data = np.array(pandas_raw)

#Remove Setosa, we hate that
iris_data = iris_data[iris_data[:,4] != 'setosa']

iris_data[:, 4] = [1 if i[4] == 'versicolor' else 0 for i in iris_data]
iris_data = iris_data.astype(float)
X = np.ones(iris_data.shape)
X[:, 1:] = iris_data[:, :4]
y = iris_data[:, 4]

w = np.ones(X.shape[1])

results = hw_x(X, w)

error = J_w(X, y, w)
print(error)
#print(w)
for i in range(2000000):
    w = grad_J_w(X, y, w, 0.01)
    #print(w)
print(w)
print(J_w(X, y, w))