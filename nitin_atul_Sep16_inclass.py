import numpy as np
import matplotlib.pyplot as plt
import time

x = np.array([
    1060, 1195, 1199, 925, 1014,
    1197, 1008, 1352, 1773, 1625,
    1827, 1325, 2120, 2700, 2659
])
# Price (y)
y = np.array([
    119000, 125000, 125000, 131000, 175000,
    175000, 187400, 194000, 200000, 225000,
    228000, 235000, 250000, 274500, 319900
])/100

def h(x, w):
    return x*w[1]+w[0]

def err(x, y, w):
    return np.mean((h(x, w) - y)**2)

def weight_update(x, y, w, alpha):
    new_0 = w[0] - alpha*(np.mean(w[1]*x -y) + w[0])
    new_1 = w[1] - alpha*(np.mean(x*((w[1]*x -y)+w[0])))
    return [new_0, new_1]
def plot_model(x, y, w):
    test_x = np.linspace(np.min(x), np.max(x), 1000)
    test_y = h(test_x, w)
    plt.plot(test_x, test_y, "r.")
    plt.plot(x, y, "b.")
    plt.xlabel("House Size(sqft)")
    plt.ylabel("Sale Price")
    plt.legend(["Lin Reg", "OG Data"])


alpha = 5e-8
w=[0,0]
n_iterations = 20
plot_model(x, y, w)
iter_error = [err(x, y, w)]
for i in range(n_iterations):
    w = weight_update(x, y, w, alpha)
    plot_model(x, y, w)
    iter_error.append(err(x, y, w))
plt.show()
print(iter_error)