import numpy as np
#My stuff ----
from sklearn.datasets import load_breast_cancer

from math import ceil
from random import shuffle, seed
from matplotlib import pyplot as plt
from math import fabs
#---------------
#load the data (you will need to change this)
data = np.load(r'classificationLabData.npz')
#Put the arrays into your workspace
print("Variables to be created:", list(data.keys()))
# Variables to be created: ['X_train', 'y_train', 'subjectID_train', 'X_test', 'subjectID_test', 'sequence_train', 'sequence_test']
locals().update(data)


#X = X_train
#Concatenate X_train and X_test
X = np.concat([X_train, X_test])

X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

X = np.hstack([np.ones((X.shape[0], 1)), X])

X_train = X[:X_train.shape[0]]
X_test = X[X_train.shape[0]:]
#y = y_train
#Now, they are numpy arrays!!!

#Some functions I used before that I prbly need here:

def hw_x(X, w):
    return 1/(1+(np.e)**(-(X@w)))

def J_w(X, y, w):
    return (-1/X.shape[0])*sum(((y*np.log(hw_x(X, w))) + ((1-y)*np.log(1-hw_x(X, w)))))

def grad_J_w(X, y, w, alpha, lam):
    return (w - ((alpha/X.shape[0])*( np.transpose(hw_x(X, w) - y)@X)) - (alpha*lam*2*w))

def plot_roc_curve(x, y, title):
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
    plt.title(title)
    plt.show()

#Preprocessing: Normalization,

#Normalization:


#Need make k-folds, run model and pick best one


print("\nOverall Model Running!!!")

w = np.zeros(X_train.shape[1])
error_over_iterations = []
print(J_w(X_train, y_train, w))

for it in range(10000):
    w = grad_J_w(X_train, y_train, w, 0.05, 0.01)
    #Don't do regularization
    #w = grad_J_w(X_train, y_train, w, 0.02, 0.01)
    #error_over_iterations.append(J_w(X_train, y_train, w))
        # print(w)
    # print(w)
print()
print(J_w(X_train, y_train, w))

'''
predictions = hw_x(X_train, w)
outcomes = y_train


# Plot Error Curve
plt.plot(range(1, len(error_over_iterations) + 1), error_over_iterations)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title(f"Error Curve For Overall")
plt.show()

false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0
for n in range(len(predictions)):
    if predictions[n] >= 0.5:
        if outcomes[n] == 1:
            true_positive += 1
        else:
            false_positive += 1
    elif predictions[n] < 0.5:
        if outcomes[n] == 0:
            true_negative += 1
        else:
            false_negative += 1
# Have our data, now print out some results!!!
print("Recall: ", true_positive / (true_positive + false_negative))
# Modify best_w, i didn't feel like taking it out of the if statement so I just made it > 0
best_w = w.copy()
best_recall = true_positive / (true_positive + false_negative)


print("Precision: ", true_positive / (true_positive + false_positive))
plot_roc_curve(predictions, outcomes, f"ROC Curve for Overall")

print("\nOverall Model")
print(best_w)
print(best_recall)
'''
predictions = hw_x(X_test, w)
np.save("scores2.npy", predictions)