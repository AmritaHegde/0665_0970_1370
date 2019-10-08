# coding: utf-8

# In[25]:

import numpy as np
from sklearn import datasets 
import seaborn.apionly as sns
import matplotlib.pyplot as plt 
sns.set(style='whitegrid', context='notebook')

# In[26]:

iris2 = sns.load_dataset('iris')

# In[27]:


def covariance (X, Y): 
    xhat=np.mean(X) 
    yhat=np.mean(Y) 
    epsilon=0
    for x,y in zip (X,Y):
        epsilon=epsilon+(x-xhat)*(y-yhat) 
    return epsilon/(len(X)-1)

# In[28]:

print (covariance ([1,3,4], [1,0,2]))

print (np.cov([1,3,4], [1,0,2]))

# In[29]:

def correlation (X, Y):
    return (covariance(X,Y)/(np.std(X,ddof=1)*np.std(Y,ddof=1)))

# In[30]:

print (correlation ([1,1,4,3], [1,0,2,2]))

print (np.corrcoef ([1,1,4,3], [1,0,2,2]))

# In[31]:

sns.pairplot(iris2, size=3.0)

# In[32]:

X=iris2['petal_width'] 
Y=iris2['petal_length']

# In[33]:

plt.scatter(X,Y)

# In[34]:

def predict(alpha, beta, x_i): 
    return beta * x_i + alpha

# In[35]:


def error(alpha, beta, x_i, y_i): #L1 return y_i - predict(alpha, beta, x_i)
    def sum_sq_e(alpha, beta, x, y): #L2
        return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))

# In[36]:

def correlation_fit(x, y):
    beta = correlation(x, y) * np.std(y, ddof=1) / np.std(x,ddof=1) 
    alpha = np.mean(y) - beta * np.mean(x)
    return alpha, beta

# In[37]:

alpha, beta = correlation_fit(X, Y) 
print(alpha)

print(beta)

# In[38]:

plt.scatter(X,Y)
xr=np.arange(0,3.5) 
plt.plot(xr,(xr*beta)+alpha)
