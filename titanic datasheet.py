#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

# Load the dataset
data_url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic = pd.read_csv('Titanic-Dataset.csv')

# Data preprocessing
titanic = titanic.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})
titanic = titanic.dropna()

# Define features and target
X = titanic.drop('Survived', axis=1).values
y = titanic['Survived'].values

# Standardize the features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (1 / m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    return cost

# Gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient
        cost_history[i] = cost_function(X, y, theta)
    
    return theta, cost_history

# Initialize parameters
theta = np.zeros(X.shape[1])
learning_rate = 0.01
iterations = 1000

# Perform gradient descent
theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

# Predict function
def predict(X, theta):
    return np.round(sigmoid(np.dot(X, theta)))

# Evaluate the model
predictions = predict(X, theta)
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)


# In[ ]:




