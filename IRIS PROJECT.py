#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


iris    = pd.read_csv("C:/Users/hp/Downloads/archive (9)/IRIS.csv")


# In[3]:


iris


# In[4]:


iris.head(5)


# In[5]:


iris.tail(5)


# In[6]:


iris.describe()


# In[7]:


iris.info()


# In[8]:


from  sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[9]:


# split dataset into features (X) and target (Y)
X = iris[['sepal_length','sepal_width','petal_length','petal_width']]
Y = iris['species']


# In[10]:


# split data into traning and testing sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)


# In[11]:


# Train RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,Y_train)


# In[12]:


# predict on test data
Y_pred = rfc.predict(X_test)


# In[13]:


# evaluate model performance
accuracy = accuracy_score(Y_test,Y_pred)
print("MODEL PERFORMANCE:")
print("-----------")
print(f"Accuracy:{accuracy*100:.2f}%")
print("CLASSIFICATION REPORT:")
print(classification_report(Y_test,Y_pred))
print("CONFUSION MATRIX:")
print(confusion_matrix(Y_test,Y_pred))


# In[ ]:




