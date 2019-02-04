#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[26]:


USAhousing = pd.read_csv('USA_Housing.csv')


# In[27]:


USAhousing.head()


# In[28]:


USAhousing.info()


# In[29]:


sns.pairplot(USAhousing)


# In[30]:


sns.heatmap(USAhousing.corr())


# In[31]:


USAhousing.columns


# In[32]:


x = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[33]:


x.head()


# In[34]:


y = USAhousing['Price']


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 101)


# In[37]:


x_train.shape


# In[38]:


from sklearn.linear_model import LinearRegression


# In[39]:


lm = LinearRegression()


# In[40]:


lm.fit(x_train, y_train)


# In[41]:


print(lm.intercept_)


# In[42]:


pd.DataFrame(lm.coef_, x.columns, columns = ['Coefs'])


# In[43]:


predict = lm.predict(x_test)


# In[51]:


plt.figure(figsize = (10,5))
plt.scatter(y_test,predict)


# In[52]:


sns.distplot((y_test-predict))


# In[53]:


#isso mostra a eficácia na escolha do modelo: curva normal


# In[54]:


from sklearn import metrics


# In[55]:


print('MAE', metrics.mean_absolute_error(y_test, predict))


# In[56]:


print('MSE', metrics.mean_squared_error(y_test, predict))


# In[58]:


print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, predict)))


# In[ ]:




