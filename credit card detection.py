#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score,classification_report,confusion_matrix

from sklearn.preprocessing import StandardScaler


# In[2]:


credit = pd.read_csv("C:/Users/hp/Downloads/archive (11)/creditcard.csv")


# In[3]:


credit


# In[4]:


credit.describe()


# In[5]:


credit.info()


# In[6]:


credit.head(5)


# In[7]:


credit.tail(5)


# In[8]:


credit.shape


# In[9]:


# drop unnecessary columns
credit = credit.drop(['Time'],axis = 1)


# In[10]:


# scale transaction amount
scaler = StandardScaler()
credit['Amount'] = scaler.fit_transform(credit[['Amount']])


# In[11]:


# split data into features (x) and target (y)
X = credit.drop('Class',axis = 1)
y = credit['Class']


# In[12]:


credit.duplicated().any()
credit = credit.drop_duplicates()
credit.shape
credit['Class'].value_counts()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[14]:


sns.countplot(credit['Class'])
plt.show()


# In[15]:


X_train,X_test,y_train ,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


# In[16]:


classifier = {
    "Logestic Regression": LogisticRegression(),
    "Decision Tree Classifier " : DecisionTreeClassifier()
}
for name , clf in classifier.items():
    print(f"\n============{name}============")
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"\n Accuracy:{accuracy_score(y_test,y_pred)}")
    print(f"\n Precision:{precision_score(y_test,y_pred)}") 
    print(f"\n Recall:{recall_score(y_test,y_pred)}")
    print(f"\n f1 Score :{ f1_score(y_test,y_pred)}")


# In[17]:


# undersampling
nor = credit[credit['Class']==0]
fraud = credit[credit['Class']==1]


# In[18]:


nor.shape


# In[19]:


fraud.shape


# In[ ]:





# In[20]:


nor_sample = nor.sample(n=473)


# In[21]:


nor_sample.shape


# In[22]:


new_credit = pd.concat([nor_sample,fraud],ignore_index = True)


# In[23]:


new_credit.head()


# In[24]:


new_credit['Class'].value_counts()


# In[25]:


X = new_credit.drop('Class',axis = 1)
y = new_credit['Class']


# In[26]:


X_train,X_test,y_train ,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)


# In[27]:


classifier = {
    "Logestic Regression": LogisticRegression(),
    "Decision Tree Classifier " : DecisionTreeClassifier()
}
for name , clf in classifier.items():
    print(f"\n============{name}============")
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"\n Accuracy:{accuracy_score(y_test,y_pred)}")
    print(f"\n Precision:{precision_score(y_test,y_pred)}") 
    print(f"\n Recall:{recall_score(y_test,y_pred)}")
    print(f"\n f1 Score :{ f1_score(y_test,y_pred)}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




