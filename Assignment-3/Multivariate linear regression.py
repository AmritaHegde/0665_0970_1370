#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualisation and plotting
import matplotlib.pyplot as plt # data plotting
import warnings


# In[2]:


# Seaborn default configuration
sns.set_style("darkgrid")


# In[3]:


# set the custom size for my graphs
sns.set(rc={'figure.figsize':(8.7,6.27)})


# In[4]:


# filter all warnings
warnings.filterwarnings('ignore') 


# In[5]:


# set max column to 999 for displaying in pandas
pd.options.display.max_columns=999 


# In[6]:


data = pd.read_csv(r'C:\Users\Chaitra\Desktop\iris.csv')


# In[7]:


data.head()


# In[11]:


mapping = {
    'Iris-setosa' : 1,
    'Iris-versicolor' : 2,
    'Iris-virginica' : 3
}
rows, col = data.shape
print("Rows : %s, column : %s" % (rows, col))


# In[12]:


X = data.drop(['Id', 'Species'], axis=1).values # Input Feature Values


# In[13]:


y = data.Species.replace(mapping).values.reshape(rows,1) # Output values


# In[14]:


X = np.hstack(((np.ones((rows,1))), X))# Adding one more column for bias


# In[15]:


np.random.seed(0) # Let's set the zero for time being


# In[16]:


theta = np.random.randn(1,5) # Setting values of theta randomly


# In[17]:


print("Theta : %s" % (theta))


# In[18]:


iteration = 10000
learning_rate = 0.003 # If you are going by formula, this is actually alpha.
J = np.zeros(iteration) # 1 x 10000 maxtix


# In[19]:


# Let's train our model to compute values of theta
for i in range(iteration):
    J[i] = (1/(2 * rows) * np.sum((np.dot(X, theta.T) - y) ** 2 ))
    theta -= ((learning_rate/rows) * np.dot((np.dot(X, theta.T) - y).reshape(1,rows), X))

prediction = np.round(np.dot(X, theta.T))

ax = plt.subplot(111)
ax.plot(np.arange(iteration), J)
ax.set_ylim([0,0.15])
plt.ylabel("Cost Values", color="Green")
plt.xlabel("No. of Iterations", color="Green")
plt.title("Mean Squared Error vs Iterations")
plt.show()


# In[20]:


ax = sns.lineplot(x=np.arange(iteration), y=J)
plt.show()


# In[21]:


ax = plt.subplot(111)

ax.plot(np.arange(1, 151, 1), y, label='Orignal value', color='red')
ax.scatter(np.arange(1, 151, 1), prediction, label='Predicted Value')

plt.xlabel("Dataset size", color="Green")
plt.ylabel("Iris Flower (1-3)", color="Green")
plt.title("Iris Flower (Iris-setosa = 1, Iris-versicolor = 2, Iris-virginica = 3)")

ax.legend()
plt.show()


# In[22]:


accuracy = (sum(prediction == y)/float(len(y)) * 100)[0]
print("The model predicted values of Iris dataset with an overall accuracy of %s" % (accuracy))


# In[ ]:




