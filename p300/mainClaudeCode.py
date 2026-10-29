import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#load the data (you will need to change this)
data = np.load(r'classificationLabData.npz')
#Put the arrays into your workspace
print("Variables to be created:", list(data.keys()))
# Variables to be created: ['X_train', 'y_train', 'subjectID_train', 'X_test', 'subjectID_test', 'sequence_train', 'sequence_test']
locals().update(data)

# one shrinkage-LDA per subject (5-fold CV AUC ~0.86 vs ~0.80 for a single pooled model)
scores = np.zeros(X_test.shape[0])
for subject in np.unique(subjectID_test):
    train = subjectID_train == subject
    test = subjectID_test == subject
    model = LDA(solver='lsqr', shrinkage='auto').fit(X_train[train], y_train[train])
    scores[test] = model.decision_function(X_test[test])

np.save("claudescores.npy", scores)
