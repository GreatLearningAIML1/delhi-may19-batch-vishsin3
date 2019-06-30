#!/usr/bin/env python
# coding: utf-8

# In[231]:


import warnings 
warnings.filterwarnings('ignore')


# ## K-Nearest-Neighbors

# KNN falls in the supervised learning family of algorithms. Informally, this means that we are given a labelled dataset consiting of training observations (x,y) and would like to capture the relationship between x and y. More formally, our goal is to learn a function h:X→Y so that given an unseen observation x, h(x) can confidently predict the corresponding output y.
# 
# In this module we will explore the inner workings of KNN, choosing the optimal K values and using KNN from scikit-learn.

# ## Overview
# 
# 1.Read the problem statement.
# 
# 2.Get the dataset.
# 
# 3.Explore the dataset.
# 
# 4.Pre-processing of dataset.
# 
# 5.Visualization
# 
# 6.Transform the dataset for building machine learning model.
# 
# 7.Split data into train, test set.
# 
# 7.Build Model.
# 
# 8.Apply the model.
# 
# 9.Evaluate the model.
# 
# 10.Finding Optimal K value
# 
# 11.Repeat 7,8,9 steps.

# ## Problem statement
# 
# ### Dataset
# 
# The data set we’ll be using is the Iris Flower Dataset which was first introduced in 1936 by the famous statistician Ronald Fisher and consists of 50 observations from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals.
# 
# **Attributes of the dataset:** https://archive.ics.uci.edu/ml/datasets/Iris
# 
# **Train the KNN algorithm to be able to distinguish the species from one another given the measurements of the 4 features.**

# ## Question 1
# 
# Import the data set and print 10 random rows from the data set

# In[232]:


import numpy as np
import pandas as pd
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from scipy.stats import zscore


# In[233]:


dev=pd.read_csv('iris (1).csv')


# In[234]:


dev.sample(10)


# ## Data Pre-processing

# ## Question 2 - Estimating missing values
# 
# *Its not good to remove the records having missing values all the time. We may end up loosing some data points. So, we will have to see how to replace those missing values with some estimated values (median) *

# In[235]:


dev.fillna(data.median(),inplace=True)
dev


# ## Question 3 - Dealing with categorical data
# 
# Change all the classes to numericals (0to2).

# In[236]:


dev['Species']=dev['Species'].replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})


# In[237]:


dev['Species']


# ## Question 4
# 
# *Observe the association of each independent variable with target variable and drop variables from feature set having correlation in range -0.1 to 0.1 with target variable.*

# In[238]:


dev1=dev.corr()
dev1['Species']
#dev1


# ## Question 5
# 
# *Observe the independent variables variance and drop such variables having no variance or almost zero variance(variance < 0.1). They will be having almost no influence on the classification.*

# In[239]:


dev.var()


# ## Question 6
# 
# *Plot the scatter matrix for all the variables.*

# In[240]:


sns.pairplot(dev)


# ## Split the dataset into training and test sets
# 
# ## Question 7
# 
# *Split the dataset into training and test sets with 80-20 ratio.*

# In[241]:


train = dev.drop(['Id','Species'],axis=1)
test = pd.DataFrame(dev['Species'])
xTrain , xTest , yTrain , yTest = train_test_split(train , test , test_size = 0.2 , random_state = 0)
yTrain


# ## Question 8 - Model
# 
# *Build the model and train and test on training and test sets respectively using **scikit-learn**. Print the Accuracy of the model with different values of **k=3,5,9**.*
# 
# **Hint:** For accuracy you can check **accuracy_score()** in scikit-learn

# In[242]:


NNH = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform',metric = 'euclidean')


# In[243]:


X_train = np.array(xTrain)
X_test = np.array(xTest)
X_train.shape
Y_train = np.array(yTrain)
Y_test = np.array(yTest)                   
Y_train


# In[244]:


NNH.fit(X_train, Y_train)


# In[245]:


Y_Predict= NNH.predict(X_test)
Y_Predict


# In[246]:


S=accuracy_score(Y_test,Y_Predict)
S


# ## Question 9 - Finding Optimal value of k.
# 
# Run the KNN with no of neighbours to be 1,3,5..19 and *Find the **optimal number of neighbours** from the above list using the Miss classification error

# Hint:
# 
# Misclassification error (MSE) = 1 - Test accuracy score. Calculated MSE for each model with neighbours = 1,3,5...19 and find the model with lowest MSE

# In[248]:


k=np.arange(1,20,2)
for neighbors in k:
    NNH = KNeighborsClassifier(n_neighbors = neighbors, weights = 'uniform',metric = 'euclidean')
    NNH.fit(X_train, Y_train)
    Y_Predict= NNH.predict(X_test)
    S=accuracy_score(Y_test,Y_Predict)
    MSE=1-accuracy_score(Y_test,Y_Predict)
    print('Misclassification Error for %d neighbors is %f'%(neighbors,MSE))
    
    
    


# ## Question 10
# 
# *Plot misclassification error vs k (with k value on X-axis) using matplotlib.*

# In[ ]:





# In[ ]:





# ### Question 11: Read the data given in bc2.csv file

# In[ ]:





# ### Question 12: Observe the no.of records in dataset and type of each feature 

# In[ ]:





# ### Question 13: Use summary statistics to check if missing values, outlier and encoding treament is necessary

# In[ ]:





# ### Check Missing Values

# In[ ]:





# ### Question 14: Check how many `?` there in Bare Nuclei feature (they are also unknown or missing values). Replace them with the top value of the describe function of Bare Nuclei feature.
# 
# #### Check include='all' parameter in describe function

# In[ ]:





# In[ ]:





# ### Question 15: Find the distribution of target variable (Class) 

# In[ ]:





# In[ ]:





# #### Plot the distribution of target variable using histogram

# In[ ]:





# ### convert the datatype of Bare Nuclei to `int`

# In[ ]:





# ### Question 16: Standardization of Data

# In[ ]:





# ### Question 17: Plot Scatter Matrix to understand the distribution of variables and check if any variables are collinear and drop one of them.

# In[ ]:





# In[ ]:





# In[ ]:





# ### Question 18: Divide the dataset into feature set and target set

# In[ ]:





# In[ ]:





# ### Divide the Training and Test sets in 70:30 

# In[ ]:





# ## Question 19 - Finding Optimal value of k
# 
# Run the KNN with no of neighbours to be 1,3,5..19 and *Find the **optimal number of neighbours** from the above list using the Mis classification error

# Hint:
# 
# Misclassification error (MSE) = 1 - Test accuracy score. Calculated MSE for each model with neighbours = 1,3,5...19 and find the model with lowest MSE

# In[ ]:





# In[ ]:





# In[ ]:





# ### Question 20: Print the optimal number of neighbors

# In[ ]:




