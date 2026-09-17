import numpy as np
import matplotlib.pyplot as plt

np.random.seed(8)
x = np.concatenate([np.random.randn(1000),np.random.randn(1000)+1.5])
y = np.concatenate([np.zeros(1000),np.ones(1000)])
#Randomize order
p = np.random.permutation(len(x))
x = x[p]
y = y[p]
#Sort order(undo the randomization, idk why you told us to do this)
p = np.argsort(x)
x=x[p]
y=y[p]
#Plot points to visualize
plt.plot(x,y,'.')
plt.xlabel("Input x-values")
plt.ylabel("Category(y--values)")
plt.show()
#Create x-values for the sort, to be plotted
dashes = np.linspace(x.min()-1e-2, x.max()+1e-2, 1000)
#Lists for values to graph later
p_detections = []
p_false_alarms = []
for dash in dashes:

    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0
    for i in range(len(x)):
        if x[i] >= dash:
            #We record as positive, check if real
            if y[i] == 1:
                true_positive += 1
            else:
                false_positive += 1
        elif x[i] < dash:
            #We record as false, check if real
            if y[i] == 0:
                true_negative += 1
            else:
                false_negative += 1

    #Now got our stuff!!!
    #Get p_detection & p_false_alarm!
    p_detection = (true_positive)/(true_positive+false_negative)
    p_false_alarm = (false_positive)/(false_positive+true_negative)
    #Add to list
    p_detections.append(p_detection)
    p_false_alarms.append(p_false_alarm)

#Plot ROC Curve as a LINE!!!
plt.plot(p_false_alarms, p_detections)
plt.xlabel("False Alarm Rate")
plt.ylabel("Detection Rate")
plt.title("ROC Curve")
plt.show()


