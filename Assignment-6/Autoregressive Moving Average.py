#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function


# In[2]:

import numpy as np


# In[3]:

from scipy import stats


# In[4]:

import pandas as pd


# In[5]:

import matplotlib.pyplot as plt


# In[6]:

import statsmodels.api as sm


# In[7]:

from statsmodels.graphics.api import qqplot


# In[8]:

print(sm.datasets.sunspots.NOTE)


# In[9]:

dta = sm.datasets.sunspots.load_pandas().data


# In[10]:

dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))


# In[11]:

del dta["YEAR"]


# In[12]:

dta.plot(figsize=(12,8));


# In[13]:

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)


# In[14]:

arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit()
print(arma_mod20.params)


# In[15]:

arma_mod30 = sm.tsa.ARMA(dta, (3,0)).fit()


# In[16]:

print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)


# In[17]:

print(arma_mod30.params)


# In[18]:

print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)


# In[19]:

sm.stats.durbin_watson(arma_mod30.resid.values)


# In[20]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = arma_mod30.resid.plot(ax=ax);


# In[21]:

resid = arma_mod30.resid


# In[22]:

stats.normaltest(resid)


# In[23]:

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)


# In[24]:

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)


# In[25]:

r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))


# In[26]:

predict_sunspots = arma_mod30.predict('1990', '2012', dynamic=True)
print(predict_sunspots)


# In[27]:

fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.ix['1950':].plot(ax=ax)
fig = arma_mod30.plot_predict('1990', '2012', dynamic=True, ax=ax, plot_insample=False)


# In[28]:

def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


# In[29]:

mean_forecast_err(dta.SUNACTIVITY, predict_sunspots)


# In[30]:

from statsmodels.tsa.arima_process import arma_generate_sample, ArmaProcess


# In[31]:

np.random.seed(1234)
# include zero-th lag
arparams = np.array([1, .75, -.65, -.55, .9])
maparams = np.array([1, .65])


# In[32]:

arma_t = ArmaProcess(arparams, maparams)


# In[34]:

arma_t.isinvertible


# In[35]:

arma_t.isstationary


# In[37]:

fig = plt.figure(figsize=(12,8))


# In[38]:

ax=fig.add_subplot(111)


# In[43]:

ax.plot(arma_t.generate_sample(50))


# In[45]:

arparams = np.array([1, .35, -.15, .55, .1])
maparams = np.array([1, .65])
arma_t = ArmaProcess(arparams, maparams)
arma_t.isstationary


# In[47]:

arma_rvs = arma_t.generate_sample(500, burnin=250, scale=2.5)


# In[48]:

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_rvs, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_rvs, lags=40, ax=ax2)


# In[49]:

arma11 = sm.tsa.ARMA(arma_rvs, (1,1)).fit()
resid = arma11.resid
r,q,p = sm.tsa.acf(resid, qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))


# In[50]:

arma41 = sm.tsa.ARMA(arma_rvs, (4,1)).fit()
resid = arma41.resid
r,q,p = sm.tsa.acf(resid, qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))


# In[51]:

macrodta = sm.datasets.macrodata.load_pandas().data
macrodta.index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
cpi = macrodta["cpi"]


# In[52]:

macrodta = sm.datasets.macrodata.load_pandas().data
macrodta.index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))
cpi = macrodta["cpi"]


# In[53]:

print(sm.tsa.adfuller(cpi)[1])


# In[ ]:




