from sklearn.datasets import load_breast_cancer
import numpy as np
from math import ceil
from random import shuffle, seed
from matplotlib import pyplot as plt
data = load_breast_cancer()
# Convert to DataFrame
X = data.data
y = data.target
#Now, they are numpy arrays!!!

#Some functions I used before that I prbly need here:

def hw_x(X, w):
    return 1/(1+(np.e)**(-(X@w)))

def J_w(X, y, w):
    return (-1/X.shape[0])*sum(((y*np.log(hw_x(X, w))) + ((1-y)*np.log(1-hw_x(X, w)))))

def grad_J_w(X, y, w, alpha):
    return (w - ((alpha/X.shape[0])*(np.transpose(hw_x(X, w) - y)@X)))

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

def get_k_fold_inds(n_observations, k):
    to_fill = list()
    for i in range(k):
        for j in range(ceil(n_observations/k)):
            to_fill.append(i+1)
        n_observations -= ceil(n_observations / k)
        k -= 1
    seed(100)#To maintain reproducibility
    shuffle(to_fill)
    return np.array(to_fill)

#Preprocessing: Normalization, then K-folds

#Normalization:
X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
X = np.hstack([np.ones((X.shape[0], 1)), X])

#Need make k-folds, run model and pick best one
k_test_stuff = get_k_fold_inds(X.shape[0], 10)
best_w = []
best_recall = 0
for i in range(1, 11):
    print("\tK-fold Number:", i)
    x_test = np.array([X[j, :] for j in range(len(k_test_stuff)) if k_test_stuff[j] == i])
    y_test = np.array([y[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] == i])
    x_training = np.array([X[j, :] for j in range(len(k_test_stuff)) if k_test_stuff[j] != i])
    y_training = np.array([y[j] for j in range(len(k_test_stuff)) if k_test_stuff[j] != i])

    #Now, I've split this into training and testing and k-folded right. Now, let's do some regression!

    w = np.zeros(X.shape[1])
    error_over_iterations = []
    print(J_w(x_training, y_training, w))
    print(J_w(x_test, y_test, w))
    # print(w)

    for it in range(400):
        w = grad_J_w(x_training, y_training, w, 0.02)
        error_over_iterations.append(J_w(x_training, y_training, w))
        # print(w)
    # print(w)
    print()
    print(J_w(x_training, y_training, w))
    print(J_w(x_test, y_test, w))

    predictions = hw_x(x_test, w)
    outcomes = y_test

    # Plot Error Curve
    plt.plot(range(1, len(error_over_iterations) + 1), error_over_iterations)
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title(f"Error Curve By Iteration for K-Fold # {i}")
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
    #In medical fields, we would prefer to maximize recall so that all potential patients are identified
    if ((true_positive / (true_positive + false_negative)) > best_recall):
        best_w = w
        best_recall = true_positive / (true_positive + false_negative)
    print("Precision: ", true_positive / (true_positive + false_positive))
    plot_roc_curve(predictions, outcomes, f"ROC Curve for testing K-Fold # {i}")

#
print("\nBest Model")
print(best_w)
print(best_recall)













































