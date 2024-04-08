#!/usr/bin/env python
# coding: utf-8

# In[8]:


import warnings
warnings.filterwarnings("ignore")


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


df = pd.read_csv("Titanic-Dataset.csv")


# In[11]:


df


# In[12]:


df.info()


# In[13]:


df.describe().T


# In[14]:


df.head()


# In[15]:


df.tail()


# In[16]:


df.shape


# In[17]:


list[df.columns]


# In[18]:


df.dtypes


# In[19]:


df.Survived.value_counts()


# In[20]:


df.Survived.describe()


# In[21]:


df.isnull().head(10)


# In[22]:


df.duplicated().sum()


# In[23]:


sns.heatmap(df.isnull(),cbar=False,cmap='Blues')


# In[24]:


df.isnull().sum()


# In[25]:


print("Missing values before imputation:")
print(df.isnull().sum())  # Check for missing values

# Methods for handling missing values:

# 1. Remove rows with missing values (not recommended for large datasets)
# df.dropna(inplace=True)
# 2. Fill with mean/median (applicable for numerical data)
# 3. Fill with a constant value (use with caution)
# 4. More sophisticated methods (e.g., interpolation for time series)

df['Age'] = df['Age'].fillna(df['Age'].mean())  # Replace with 'median' for median imputation
df['Cabin'] = df['Cabin'].fillna('missing')

print("\nMissing values after imputation:")
print(df.isnull().sum())  # Check again for missing values


# In[26]:


sns.heatmap(df.isnull(),cbar=False,cmap='Blues')


# In[27]:



sns.pairplot(df, hue='Survived', height=3)
plt.show()


# In[28]:


sns.FacetGrid(df, hue='Survived', height=7).map(sns.distplot, 'Age').add_legend()


# In[29]:


x = df[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']
# Splitting data into training and testing sets
def train_test_split_custom(x, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    test_indices = indices[:int(test_size * len(x))]
    train_indices = indices[int(test_size * len(x)):]
    
    x_train = x.iloc[train_indices]
    y_train = y.iloc[train_indices]
    x_test = x.iloc[test_indices]
    y_test = y.iloc[test_indices]
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split_custom(x, y, test_size=0.2, random_state=30)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[30]:


X = df[['Fare', 'Age']].values
Y = df['Survived'].values
X


# In[31]:


Y


# In[32]:


intercept = np.ones((X.shape[0], 1))
X = np.concatenate((intercept, X), axis=1)
#Normalize the features-
X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
X


# In[33]:


weights = np.zeros(X.shape[1])
#Sigmoid function- 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Define the logistic regression function

def logistic_regression(X, weights):
    return sigmoid(np.dot(X, weights))

#Define the loss function (binary cross-entropy)

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

#Define the gradient descent function

def gradient_descent(X, y, weights, learning_rate, num_iterations):
    for i in range(num_iterations):
        # Calculate predictions
        y_pred = logistic_regression(X, weights)
        
        # Calculate gradients
        dw = np.dot(X.T, (y_pred - y)) / len(y)
        
        # Update weights
        weights -= learning_rate * dw
         # Print loss every 100 iterations
        if i % 100 == 0:
            loss = binary_cross_entropy(y, y_pred)
            print(f'Iteration {i}, Loss: {loss}')
    
    return weights


# In[34]:


#Hyperparameters-

learning_rate = 0.01
num_iterations = 1000
#Train the model using gradient descent-

weights = gradient_descent(X, y, weights, learning_rate, num_iterations)


# In[35]:


#Make predictions-

y_pred = np.round(logistic_regression(X, weights))
#Calculate accuracy-

accuracy = np.mean(y_pred == y)
print(f'Accuracy: {accuracy}')


# # DECISION TREE

# In[36]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the Decision Tree classifier
dt_classifier.fit(x_train, y_train)

# Predictions
dt_predictions = dt_classifier.predict(x_test)

# Calculate accuracy
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f'Decision Tree Accuracy: {dt_accuracy}')

# Classification Report
print(classification_report(y_test, dt_predictions))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, dt_predictions), annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Decision Tree Confusion Matrix')
plt.show()


# # SVM

# In[37]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the SVM classifier
svm_classifier.fit(x_train, y_train)

# Predictions
svm_predictions = svm_classifier.predict(x_test)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f'SVM Accuracy: {svm_accuracy}')

# Classification Report
print(classification_report(y_test, svm_predictions))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, svm_predictions), annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('SVM Confusion Matrix')
plt.show()


# In[ ]:




