#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/hp/Downloads/archive (8)/Titanic-Dataset.csv")


# In[3]:


df


# In[4]:


# look at top 5 pasenger according to passengerla
df.head(5)


# In[5]:


df.describe()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df['Survived'].value_counts()


# In[9]:


# passeger survived according  to  pclass
sns.countplot(x=df['Survived'],hue = df['Pclass'])


# In[10]:


df['Age'].value_counts()


# In[11]:


df['Age'].info()


# In[12]:


# drop some coloumn
df.drop(['Name','Cabin','Ticket'],axis=1,inplace = True)


# In[13]:


df


# In[14]:


# convert categorical variables to numerical
df['Sex'] = df['Sex'].map({'male':0,'female':1})
df


# In[15]:


df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2})
df


# In[16]:


# Handle missing value
df['Age'] = df['Age'].fillna(df['Age'].median())


# In[17]:


# split data
X = df.drop(['Survived'],axis = 1)
Y = df['Survived']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


df.dropna(inplace=True)


# In[19]:


from sklearn.impute import SimpleImputer
imputor = SimpleImputer(strategy='mean')
X_filled = X.interpolate()
 


# In[20]:


from sklearn.impute import KNNImputer


# In[ ]:





# In[ ]:





# In[21]:


# RANDOM FOREST CLASSIFIER
rfc = RandomForestClassifier(n_estimators = 100,random_state = 42) 
rfc.fit(X_train,Y_train)
Y_pred = rfc.predict(X_test)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import  classification_report
from sklearn.metrics import *
# evaluate model
print("Accuracy:",accuracy_score(Y_test,Y_pred))
print("Classification Report:")
print(classification_report(Y_test,Y_pred))
print("confusion Matrix:")
print(confusion_matrix(Y_test,Y_pred))


# In[ ]:


# visualize Results
sns.set()
sns.countplot(x='Age',hue='Survived',data=df)
plt.title("SURVIVAL BY AGE")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




