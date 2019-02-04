#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


from plotly import __version__


# In[20]:


print(__version__)


# In[21]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[22]:


import cufflinks as cf


# In[23]:


init_notebook_mode(connected = True)


# In[24]:


cf.go_offline()


# In[33]:


df = pd.DataFrame(np.random.randn(100, 4), columns = 'A B C D'.split())


# In[34]:


df.head()


# In[35]:


df2 = pd.DataFrame({'Categoria':['A', 'B', 'C'], 'Valores':[32, 43, 50]})


# In[36]:


df2


# In[37]:


df.plot(kind = 'scatter', x = 'A', y = 'B')


# In[40]:


df.iplot(kind = 'scatter', x = 'A', y = 'B', mode = 'markers')


# In[41]:


df2.iplot(kind ='bar', x = 'Categoria', y = 'Valores')


# In[45]:


df.iplot(kind ='bar')


# In[46]:


df.iplot(kind = 'box')


# In[47]:


df3 = pd.DataFrame({'x':[1, 2, 3, 4, 5], 'y':[10, 20, 30, 40, 50], 'z':[5, 4, 3, 2, 1]})


# In[48]:


df3


# In[50]:


df3.iplot(kind = 'surface', colorscale = 'rdylbu')


# In[51]:


df[['A', 'B']].iplot(kind = 'spread')


# In[56]:


df[['A', 'B']].iplot(kind = 'hist', bins = 50)


# In[58]:


df.iplot(kind = 'bubble', x = 'A', y = 'B', size = 'C')


# In[59]:


df.scatter_matrix()


# In[ ]:




