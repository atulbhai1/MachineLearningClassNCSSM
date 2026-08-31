import numpy as np
import matplotlib.pyplot as plt


def weight_calc(X, y):
    return np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y))
x = np.linspace(0, 10)#np.random.rand(20)*10
y = np.sin(x)#*np.random.normal(20, size=20)*0.2
plt.plot(x, y, "x")

X = np.concatenate([np.ones(x.shape)[:, None], x[:,None]], axis=1)

w = weight_calc(X, y)

plt.plot(x, np.dot(X, w), ".")

X = np.concatenate([np.ones(x.shape)[:, None], x[:,None], x[:, None]**2, x[:,None]**3], axis=1)
w = weight_calc(X, y)
print(w)
plt.plot(x, np.dot(X, w), ".")

X = np.concatenate([np.ones(x.shape)[:, None], x[:,None], x[:, None]**2, x[:,None]**3, x[:, None]**4], axis=1)
w = weight_calc(X, y)
print(w)
plt.plot(x, np.dot(X, w), ".")

X = np.concatenate([np.ones(x.shape)[:, None], x[:,None], x[:, None]**2, x[:,None]**3, x[:, None]**4], axis=1)
w = weight_calc(X, y)
print(w)
plt.plot(x, np.dot(X, w), ".")

temp = [np.ones(x.shape)[:,None]]
for i in range(1, 11):
    temp.append(x[:, None]**i)
    X = np.concatenate(temp,
                       axis=1)
    w = weight_calc(X, y)
    print(w)
    plt.plot(x, np.dot(X, w), ".")

plt.show()


