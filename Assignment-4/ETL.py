#!/usr/bin/env python
# coding: utf-8

# In[28]:


import mysql.connector


# In[29]:


from mysql.connector import Error


# In[30]:


import json


# In[31]:


import collections


# In[32]:


import pandas as pd


# In[33]:


conn=mysql.connector.connect(host='localhost',database='etl',user='root',password='')


# In[34]:


cursor=conn.cursor()


# In[35]:


cursor.execute("SELECT * from user_device")


# In[36]:


rows=cursor.fetchall()


# In[40]:


columns=['use_id','user_id','platform','platform_version','device','use_type_id']


# In[41]:


df1=pd.DataFrame(rows,columns=columns)


# In[42]:


df1.head()


# In[43]:


header=['Retail_branding','Marketing_Name','Device','Model']


# In[46]:


connn=mysql.connector.connect(host='localhost',database='etl',user='root',password='')


# In[48]:


cursorn=connn.cursor()


# In[49]:


cursorn.execute('SELECT * from android_devices')


# In[50]:


android=cursorn.fetchall()


# In[51]:


df2=pd.DataFrame(android,columns=header)


# In[52]:


df2.head()


# In[53]:


result=pd.merge(df1,
                df2[['Retail_branding','Model']],
                left_on='device',
                right_on='Model',
                how='left')


# In[54]:


final=pd.DataFrame(result)


# In[55]:


final.head()


# In[ ]:




