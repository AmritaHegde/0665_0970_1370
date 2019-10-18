#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualisation and plotting
import matplotlib.pyplot as plt # data plotting
import warnings


# In[30]:


# Seaborn default configuration
sns.set_style("darkgrid")


# In[31]:


# set the custom size for my graphs
sns.set(rc={'figure.figsize':(8.7,6.27)})


# In[32]:


# filter all warnings
warnings.filterwarnings('ignore') 


# In[33]:


# set max column to 999 for displaying in pandas
pd.options.display.max_columns=999 


# In[34]:


data = pd.read_csv(r'C:\Users\Chaitra\Desktop\iris.csv')


# In[35]:


data.head()


# In[36]:


data.info()


# In[37]:


data.describe()


# In[38]:


data['Species'].value_counts()


# In[39]:


rows, col = data.shape
print("Rows : %s, column : %s" % (rows, col))


# In[40]:


snsdata = data.drop(['Id'], axis=1)
g = sns.pairplot(snsdata, hue='Species', markers='x')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)


# In[41]:


sns.violinplot(x='SepalLengthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='SepalWidthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='PetalLengthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()
sns.violinplot(x='PetalWidthCm', y='Species', data=data, inner='stick', palette='autumn')
plt.show()


# In[ ]:




