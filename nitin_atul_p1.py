"""
Created By: Atul Nitin
Assignment Number: P0
Created Date: Sep. 3rd, 2026
Last Modified: Sep. 3rd, 2026
Requirements:
- pandas
- numpy
- matplotlib
"""
from random import randint

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linear_regression_weight_calc(X, y):
    return np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y))

def calc_error(X, y, w):
    return np.mean(np.square(np.dot(X, w) - y))

def get_k_fold_inds(n_observations, k):
    to_fill = list()
    for ind in range(n_observations):
        to_fill.append(randint(1, k+1))
    return np.array(to_fill)

#Drag data into pandas because csv reader keeps the top columns, and I don't like that
pandas_raw = pd.read_csv("p1_mpg.csv")

#I really like pandas, but here is my reluctantly created numpy code 😢

#Turn into Array for numpy stuff
mpg_data = np.array(pandas_raw)
displacement = mpg_data[:, 1]
cylinders = mpg_data[:, 2]
weight = mpg_data[:, 3]
acceleration = mpg_data[:, 4]
mpg = mpg_data[:, 5]

#For mpg_estimator function
X = np.ones((displacement.__len__(), 5))
for i in range(len(displacement)):
    X[i, 1] = displacement[i]
    X[i, 2] = cylinders[i]
    X[i, 3] = weight[i]
    X[i, 4] = acceleration[i]
y = mpg
mpg_estimator_weights = linear_regression_weight_calc(X, y)

def mpg_estimator(displacement, cylinders, weight, acceleration):
    return mpg_estimator_weights[0] + mpg_estimator_weights[1]*displacement + mpg_estimator_weights[2]*cylinders + mpg_estimator_weights[3]*weight + mpg_estimator_weights[4]*acceleration
#print(mpg_data)
'''
n_points = 7

t = np.linspace(0, 10, n_points)

d = 0.5 * 9.98 * t ** 2 + np.random.normal(0, size=t.shape[0]) * 40

plt.plot(t, d, 'o')

plt.xlabel('time')

plt.ylabel('distance')

n_poly = 6

T = np.ones(t.shape[0])[:, None]

for p in range(n_poly):
    T = np.concatenate([T, t[:, None] ** (p + 1)], axis=1)

w = linear_regression_weight_calc(T, d)

t_test = np.linspace(0, 10, 1000)

T_test = np.ones(t_test.shape[0])[:, None]

for p in range(n_poly):
    T_test = np.concatenate([T_test, t_test[:, None] ** (p + 1)], axis=1)

plt.plot(t_test, np.dot(T_test, w), '.')
plt.show()
'''

#Do regression for all the attributes, do k-folds

print("Cylinders:")
k_test_stuff = get_k_fold_inds(len(cylinders), 10)
for i in range(1, 11):
    print("\tK-fold Number:", i)
    x_test = [cylinders[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] == i]
    y_test = [mpg[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] == i]
    x_training = [cylinders[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] != i]
    y_training = [mpg[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] != i]
    X_training = np.ones((x_training.__len__(), 2))
    for i in range(len(x_training)):
        X_training[i, 1] = x_training[i]
    X_test = np.ones((x_test.__len__(), 2))
    for i in range(len(x_test)):
        X_test[i, 1] = x_test[i]
    training_weights = linear_regression_weight_calc(X_training, y_training)
    print("\t\tWeights[w0 w1]:")
    print("\t\t\t", training_weights, sep="")
    print("\t\tError:")
    print("\t\t\t", calc_error(X_test, y_test, training_weights), sep="")
    print()

print("Displacement:")
k_test_stuff = get_k_fold_inds(len(displacement), 10)
for i in range(1, 11):
    print("\tK-fold Number:", i)
    x_test = [displacement[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] == i]
    y_test = [mpg[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] == i]
    x_training = [displacement[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] != i]
    y_training = [mpg[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] != i]
    X_training = np.ones((x_training.__len__(), 2))
    for i in range(len(x_training)):
        X_training[i, 1] = x_training[i]
    X_test = np.ones((x_test.__len__(), 2))
    for i in range(len(x_test)):
        X_test[i, 1] = x_test[i]
    training_weights = linear_regression_weight_calc(X_training, y_training)
    print("\t\tWeights[w0 w1]:")
    print("\t\t\t", training_weights, sep="")
    print("\t\tError:")
    print("\t\t\t", calc_error(X_test, y_test, training_weights), sep="")
    print()

print("Weight:")
k_test_stuff = get_k_fold_inds(len(weight), 10)
for i in range(1, 11):
    print("\tK-fold Number:", i)
    x_test = [weight[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] == i]
    y_test = [mpg[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] == i]
    x_training = [weight[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] != i]
    y_training = [mpg[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] != i]
    X_training = np.ones((x_training.__len__(), 2))
    for i in range(len(x_training)):
        X_training[i, 1] = x_training[i]
    X_test = np.ones((x_test.__len__(), 2))
    for i in range(len(x_test)):
        X_test[i, 1] = x_test[i]
    training_weights = linear_regression_weight_calc(X_training, y_training)
    print("\t\tWeights[w0 w1]:")
    print("\t\t\t", training_weights, sep="")
    print("\t\tError:")
    print("\t\t\t", calc_error(X_test, y_test, training_weights), sep="")
    print()

print("Acceleration:")
k_test_stuff = get_k_fold_inds(len(acceleration), 10)
for i in range(1, 11):
    print("\tK-fold Number:", i)
    x_test = [acceleration[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] == i]
    y_test = [mpg[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] == i]
    x_training = [acceleration[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] != i]
    y_training = [mpg[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] != i]
    X_training = np.ones((x_training.__len__(), 2))
    for i in range(len(x_training)):
        X_training[i, 1] = x_training[i]
    X_test = np.ones((x_test.__len__(), 2))
    for i in range(len(x_test)):
        X_test[i, 1] = x_test[i]
    training_weights = linear_regression_weight_calc(X_training, y_training)
    print("\t\tWeights[w0 w1]:")
    print("\t\t\t", training_weights, sep="")
    print("\t\tError:")
    print("\t\t\t", calc_error(X_test, y_test, training_weights), sep="")
    print()

#---Part 4: Choice 2

#Displacement & Cylinders
print("Displacement + Cylinder:")
X = np.ones((displacement.__len__(), 3))
for i in range(len(displacement)):
    X[i, 1] = displacement[i]
    X[i, 2] = cylinders[i]
y = mpg
training_weights = linear_regression_weight_calc(X, y)
print("\t\tWeights[w0 w1 w2]:")
print("\t\t\t", training_weights, sep="")
print("\t\tError:")
print("\t\t\t", calc_error(X, y, training_weights), sep="")
print()

#Displacement & Cylinders & Weight
print("Displacement + Cylinder + Weight:")
X = np.ones((displacement.__len__(), 4))
for i in range(len(displacement)):
    X[i, 1] = displacement[i]
    X[i, 2] = cylinders[i]
    X[i, 3] = weight[i]
y = mpg
training_weights = linear_regression_weight_calc(X, y)
print("\t\tWeights[w0 w1 w2 w3]:")
print("\t\t\t", training_weights, sep="")
print("\t\tError:")
print("\t\t\t", calc_error(X, y, training_weights), sep="")
print()

#Displacement & Cylinders & Weight
print("Displacement + Cylinder + Weight + Acceleration:")
X = np.ones((displacement.__len__(), 5))
for i in range(len(displacement)):
    X[i, 1] = displacement[i]
    X[i, 2] = cylinders[i]
    X[i, 3] = weight[i]
    X[i, 4] = acceleration[i]
y = mpg
training_weights = linear_regression_weight_calc(X, y)
print("\t\tWeights[w0 w1 w2 w3 w4]:")
print("\t\t\t", training_weights, sep="")
print("\t\tError:")
print("\t\t\t", calc_error(X, y, training_weights), sep="")
print()

#Part 5
print("For measurement 2, the predicted mpg is")
print(mpg_estimator(displacement[1], cylinders[1], weight[1], acceleration[1]))
print("In reality, it was")
print(mpg[1])

#Part 6
'''
I chose to make a model which uses all of the attributes because in general,
 using more parameters brought down the error in step 4. I used displacement,
 weight, acceleration, and cylinders as they were all of the features available.
 This model(the one with the most parameters) had the lowest error of any of
 the multi-parameter functions at 16.034996173707945. There were models made by
 k-folds in step 3 on some runs that had lower errors but I ignored those as
 those were the error scores for just the test data that had been randomly
 allotted, so it could just be that the model got really lucky with which data
 was the test data.'''