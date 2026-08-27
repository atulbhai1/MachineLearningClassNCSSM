import numpy as np
import matplotlib.pyplot as plt

x = [
    1060, 1195, 1199, 925, 1014,
    1197, 1008, 1352, 1773, 1625,
    1827, 1325, 2120, 2700, 2659
]
# Price (y)
y = [
    119000, 125000, 125000, 131000, 175000,
    175000, 187400, 194000, 200000, 225000,
    228000, 235000, 250000, 274500, 319900
]

y = np.array(y)
x=np.array(x)

n = len(x)

w0 = 100000
w1 = 50

def error(x, y, n, w0, w1):
    J = (1/n) * sum((((w1*x) + w0) - y)**2)
    return J

#Make an estimate

#Try for first 2 values
x1 = x[:2]
y1 = y[:2]

119000 - 1060*44.4444

min_error = 9.9*10**10
for i in range(8500, 9600):
    i = i/100
    #i is w1 ig???
    for j in range(50000,70000):
        #j is w0 ig???
        if error(x, y, n, j, i) <= min_error:
            min_error = error(x, y, n, j, i)
            #print(min_error)
            w0 = j
            w1 = i

print(w0, w1)

print(error(x, y, n, w0, w1))

plt.plot(x, y, 'o')
#Generate points
x_show = [1000, 2700]
y_show = [(w1*1000+w0), w1*2700+w0]
plt.plot(x_show, y_show)
plt.show()
