
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

objects = ('BJP',
'SS',
'JDU',
'INC',
'DMK',
'TMC',
'YSRCP',
'BJD',
'BSP',
'OTH')
y_pos = np.arange(len(objects))
performance = [303,
18,
16,
52,
23,
22,
22,
12,
10,
96              
]

plt.bar(y_pos, performance, align='center')
plt.xticks(y_pos, objects)
plt.xlabel('Parties')
plt.ylabel('No_of_seats_won')
plt.title('Lok Sabha election result 2019')

plt.show()


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


plt.hist(performance)

plt.show()


# In[11]:


fig1, ax1 = plt.subplots()
ax1.pie(performance, labels=objects, 
        shadow=True, startangle=90)
ax1.axis('equal') 

plt.show()


# In[10]:


plt.scatter(objects, performance, marker='*',color='red');

