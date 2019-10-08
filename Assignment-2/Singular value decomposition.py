
# coding: utf-8

# In[3]:


import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) 

df


# In[4]:


X = df.iloc[:,0:4].values
y = df.iloc[:,4].values


# In[5]:


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)


# In[9]:


cor_mat1 = np.corrcoef(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# In[28]:


import numpy as np
u,s,v = np.linalg.svd(X_std.T)


# In[25]:


u


# In[14]:


s


# In[22]:


v

