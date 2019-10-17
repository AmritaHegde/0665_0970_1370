#!/usr/bin/env python
# coding: utf-8

# In[34]:

# line plot of time series
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from pandas import datetime
import numpy
# load dataset
series = read_csv(r'C:\Users\Chaitra\Desktop\dataset2.csv', header=0, index_col=0)
# display first few rows
print(series.head(20))


# In[35]:

# line plot of dataset
series.plot()
pyplot.show()


# In[36]:

# split the dataset
split_point = len(series) - 7
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', index=False)
validation.to_csv('validation.csv', index=False)


# In[37]:

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)


# In[38]:

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


# In[39]:

X = series.values


# In[40]:

days_in_year = 365


# In[41]:

differenced = difference(X, days_in_year)


# In[42]:

model = ARIMA(differenced, order=(7,0,1))


# In[43]:

model_fit = model.fit(disp=0)


# In[44]:

print(model_fit.summary())


# In[45]:

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


# In[46]:

forecast = model_fit.forecast()[0]


# In[47]:

forecast = inverse_difference(X, forecast, days_in_year)


# In[48]:

print('Forecast: %f' % forecast)


# In[49]:

# one-step out of sample forecast
start_index = len(differenced)
end_index = len(differenced)
forecast = model_fit.predict(start=start_index, end=end_index)


# start_index='1990-12-25'

# In[50]:

print(forecast)


# In[51]:

# multi-step out-of-sample forecast
forecast = model_fit.forecast(steps=7)[0]


# In[52]:

# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1





