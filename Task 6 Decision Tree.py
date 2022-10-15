#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('D:/The Spark foundation/Iris.csv')


# In[3]:


df.head(10)


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.columns


# In[7]:


df.isna().sum()


# In[8]:


df.corr().abs()


# In[9]:


df.groupby(['Species']).count()


# In[10]:


df.hist(bins = 20,figsize = (15,10));


# In[11]:


plt.figure(figsize=(7,5))
sns.heatmap(df.corr(), annot=True,cmap="RdYlGn")
plt.show()


# In[12]:


sns.countplot(x='Species',data=df)
plt.title('Species')
plt.show()


# In[13]:


sns.set(style = 'whitegrid')
sns.stripplot(x ='Species',y = 'SepalLengthCm',data = df);
plt.title('Iris Dataset')
plt.show()


# In[14]:


sns.set(style = 'whitegrid')
sns.scatterplot(x ='PetalLengthCm',y = 'PetalWidthCm',hue="Species",data = df);
plt.title('Iris Dataset')
plt.show()


# In[15]:


sns.set(style = 'whitegrid')
sns.stripplot(x ='Species',y = 'PetalLengthCm',data = df);
plt.title('Iris Dataset')
plt.show()


# In[16]:


sns.set(style = 'whitegrid')
sns.scatterplot(x ='SepalLengthCm',y = 'SepalWidthCm',hue="Species",data = df);
plt.title('Iris Dataset')
plt.show()


# In[17]:


sns.boxplot(x='Species',y='PetalLengthCm',data=df)
plt.title("Iris Dataset")
plt.show()


# In[18]:


sns.boxplot(x='Species',y='PetalWidthCm',data=df)
plt.title("Iris Dataset")
plt.show()


# In[19]:


y = df['Species']
X = df.drop(['Species',"Id"],axis=1)
X.shape, y.shape


# In[20]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, X_test.shape


# In[21]:


from sklearn.tree import DecisionTreeClassifier


# In[22]:


decision = DecisionTreeClassifier()


# In[23]:


decision.fit(X_train, y_train)


# In[24]:


y_test_tree = decision.predict(X_test)
y_train_tree = decision.predict(X_train)


# In[25]:


print(decision.score(X_test,y_test_tree))
print(decision.score(X_train,y_train_tree))


# In[26]:


from sklearn.metrics import classification_report
print(classification_report(y,decision.predict(X)))


# In[27]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y,decision.predict(X),labels=df['Species'].unique()))


# In[28]:


from sklearn import tree
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
tree.plot_tree(decision,feature_names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'] ,
               class_names = df['Species'].unique(), filled=True)
plt.show()


# In[ ]:




