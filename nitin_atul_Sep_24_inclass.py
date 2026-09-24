import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def hw_x(X, w):
    return 1/(1+(np.e)**(-(X@w)))
def J_w(X, y, w):
    return (-1/X.shape[0])*sum(((y*np.log(hw_x(X, w))) + ((1-y)*np.log(1-hw_x(X, w)))))
def grad_J_w(X, y, w, alpha):
    return (w - ((alpha/X.shape[0])*(np.transpose(hw_x(X, w) - y)@X)))
raw = pd.read_csv("student.csv", sep=";")
raw = np.array(raw)
new = np.zeros((len(raw), 29))
for i in range(len(raw)):
    if raw[i][0] == "GP":
        new[i][0] = 0
    else:
        new[i][0] = 1
    if raw[i][1] == "F":
        new[i][1] = 0
    else:
        new[i][1] = 1
    if raw[i][3] == "R":
        new[i][3] = 0
    else:
        new[i][3] = 1
    if raw[i][4] == "LE3":
        new[i][4] = 0
    else:
        new[i][4] = 1
    if raw[i][5] == "T":
        new[i][5] = 0
    else:
        new[i][5] = 1
    new[i][6] = raw[i][6]
    new[i][7] = raw[i][7]
    #Skip 8 & 9 in new, keep 8th index for next valid one
    new[i][8] = raw[i][12]
    new[i][9] = raw[i][13]
    new[i][10] = raw[i][14]
    if raw[i][15] == "yes":
        new[i][11] = 0
    else:
        new[i][11] = 1
    if raw[i][16] == "yes":
        new[i][12] = 0
    else:
        new[i][12] = 1
    if raw[i][17] == "yes":
        new[i][13] = 0
    else:
        new[i][13] = 1
    if raw[i][18] == "yes":
        new[i][14] = 0
    else:
        new[i][14] = 1
    if raw[i][19] == "yes":
        new[i][15] = 0
    else:
        new[i][15] = 1
    if raw[i][20] == "yes":
        new[i][16] = 0
    else:
        new[i][16] = 1
    if raw[i][21] == "yes":
        new[i][17] = 0
    else:
        new[i][17] = 1
    if raw[i][22] == "yes":
        new[i][18] = 0
    else:
        new[i][18] = 1
    new[i][19] = raw[i][23]
    new[i][20] = raw[i][24]
    new[i][21] = raw[i][25]
    new[i][22] = raw[i][26]
    new[i][23] = raw[i][27]
    new[i][24] = raw[i][28]
    new[i][25] = raw[i][29]
    new[i][26] = raw[i][30]
    new[i][27] = raw[i][31]
    if raw[i][32] >= 10:
        new[i][28] = 0
    else:
        new[i][28] = 1
X = np.ones(new.shape)
X[:, 1:] = new[:, :28]
y = raw[:, 28]
w = np.ones(new.shape[1])

X = X.astype(float)
y = y.astype(float)

for i in range(200):
    w = grad_J_w(X, y, w, 0.01)

print(w)
print(hw_x(X[382], w))