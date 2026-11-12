# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve
import math

# Load MNIST dataset using scikit-learn
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

# %%


# Convert NumPy arrays to PyTorch tensors


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
chosen_activation = nn.ReLU

# Build your own neural network
# This code won't run as is you need to decide what layers you want and how many nodes etc.
# Make some new layers that you can use for forward passes:
# self.layer = nn.Linear(n_nodes_in, n_nodes_out)
# Add those new layers to the forward function to build them into your model. T
# 5. Define MLP Architecture (784 inputs -> 100 hidden units -> 10 outputs)


dropoutRate = 0.049
class MLP(nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                self.network = nn.Sequential(nn.Linear(784, 784//2, bias=True), chosen_activation(), nn.Dropout(dropoutRate), nn.Linear(784//2, 784//2, bias=True), chosen_activation(), nn.Dropout(dropoutRate), nn.Linear(784//2, 784//2, bias=True), chosen_activation(), nn.Dropout(dropoutRate), nn.Linear(784//2, 784//2, bias=True), chosen_activation(), nn.Dropout(dropoutRate), nn.Linear(784//2, 784//2, bias=True), chosen_activation(), nn.Dropout(dropoutRate), nn.Linear(784//2, 10, bias=True))

            def forward(self, x):
                return self.network(x)


model = MLP()

        # Define loss function and optimizer (simple SGD)
error_func = nn.CrossEntropyLoss()  # choose a loss function
optimizer = optim.SGD(model.parameters(), lr= 0.01)  # Set learning rate for our gradient descent

epochs = 10  # Number of times training uses each observation in the training set
batch_size = 32  # Number of observations per batch (should be some fraction of the training dataset)

n_train_samples = X_train_tensor.shape[0]

for epoch in range(epochs):  # Looping over the number of epoch
            for i in range(0, n_train_samples, batch_size):
                # Get a mini-batch
                inputs = X_train_tensor[i:i + batch_size]
                targets = y_train_tensor[i:i + batch_size]

                # Zero the gradients - necessary for the start of each batch
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Calculate loss
                loss = error_func(outputs, targets)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                if (i // batch_size) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i // batch_size + 1}/{n_train_samples // batch_size + (1 if n_train_samples % batch_size != 0 else 0)}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
            test_outputs = model(X_test_tensor)
            loss = error_func(test_outputs, y_test_tensor)
            train_outputs = model(X_train_tensor)
            loss_train = error_func(train_outputs, y_train_tensor)
            predictions = torch.argmax(test_outputs, dim=1)
            acc = accuracy_score(y_test_tensor.numpy(), predictions.numpy())
            # accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
            print(f'\tAccuracy on the test set: {acc * 100:.2f}%')
        # %%
