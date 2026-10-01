import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data/255
y = mnist.target.astype(int)
y = (y==5)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=10000,random_state=42,stratify=y)

pca = PCA(n_components=2)
pca_out_train = pca.fit_transform(X_train)
pca_out_test = pca.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=1)
lda_out_train = lda.fit_transform(X_train,y_train)
lda_out_test = lda.transform(X_test)


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

print("PCA!!!!")
X = pca_out_train
w = np.ones(X.shape[1])
error_over_iterations = []
print(J_w(X, y_train, w))
print(J_w(pca_out_test, y_test, w))
#print(w)
for i in range(1000):
    w = grad_J_w(X, y_train, w, 0.04)
    error_over_iterations.append(J_w(X, y_train, w))
    #print(w)
#print(w)
print()
print(J_w(X, y_train, w))
print(J_w(pca_out_test, y_test, w))
predictions = hw_x(pca_out_test, w)
outcomes = y_test

#Plot Error Curve
plt.plot(list(range(1, 1001)), error_over_iterations)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title("Error Curve By Iteration for PCA")
plt.show()

false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        if outcomes[i] == 1:
            true_positive += 1
        else:
            false_positive += 1
    elif predictions[i] < 0.5:
        if outcomes[i] == 0:
            true_negative += 1
        else:
            false_negative += 1
#Have our data, now print out some results!!!
print("Recall: ", true_positive/(true_positive+false_negative))
print("Precision: ", true_positive/(true_positive+false_positive))
plot_roc_curve(predictions, outcomes, "ROC Curve for PCA testing")

print()

print("LDA!!!!")
X = lda_out_train
w = np.ones(X.shape[1])
error_over_iterations = []
print(J_w(X, y_train, w))
print(J_w(lda_out_test, y_test, w))
#print(w)
for i in range(200):
    w = grad_J_w(X, y_train, w, 0.02)
    error_over_iterations.append(J_w(X, y_train, w))
    #print(w)
#print(w)
print()
print(J_w(X, y_train, w))
print(J_w(lda_out_test, y_test, w))

#ROC Curve for LDA testing, ig?
predictions = hw_x(lda_out_test, w)
outcomes = y_test

#Plot Error Curve
plt.plot(list(range(1, 201)), error_over_iterations)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title("Error Curve By Iteration for LDA")
plt.show()

false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        if outcomes[i] == 1:
            true_positive += 1
        else:
            false_positive += 1
    elif predictions[i] < 0.5:
        if outcomes[i] == 0:
            true_negative += 1
        else:
            false_negative += 1
#Have our data, now print out some results!!!
print("Recall: ", true_positive/(true_positive+false_negative))
print("Precision: ", true_positive/(true_positive+false_positive))

print("")
print("ROC Curve for LDA testing")

plot_roc_curve(predictions, outcomes, "ROC Curve for LDA testing")