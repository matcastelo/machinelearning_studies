#!/usr/bin/env python
# coding: utf-8

# # 1. Criação de vetores e matrizes no numpy

# In[5]:


import numpy as np

#criando matriz de zeros
np.zeros((4,4))


# In[6]:


#criando matriz de um's
np.ones((3,3))


# In[8]:


#matriz identidade 
np.eye(5)


# In[9]:


# O terceiro argumento é o espacamento
np.linspace (0, 10, 3)


# In[10]:


np.arange(0, 10)


# In[11]:


np.arange(0,100,5)


# In[12]:


# rand() criando um veto de numeros floats aleatórios (são tirados de uma distribuição uniforme)
np.random.rand(10)


# In[17]:


# criando uma matriz com valores aleatórios
np.random.rand(5,6)


# In[18]:


#randn() tira números de uma distribuição normal
np.random.randn(3,3)


# In[19]:


#randint() cria números inteiros aleatórios a partir de um intervalo
np.random.randint(0,100, 10)


# In[23]:


arr = np.random.rand(25)
arr


# In[27]:


#utilizando reshape para transformar o vetor em matriz
arr.reshape((5,5))


# In[28]:


#shape é um atributo e não metódo... método tem ()
arr.shape


# In[29]:


arr.max()


# In[30]:


arr.min()


# # 2. Indexação e fatiamento

# In[32]:


#criando vetor no intervalo de 0 a 30 de 3 em 3
array = np.arange(0,30,3)

array


# In[33]:


array[4]


# In[34]:


array[3:5]


# In[36]:


#criando matriz 5x10
arr = np.arange(0,50).reshape(5,10)

arr


# In[39]:


#puxando determinado intervalo [linha][coluna], 
arr[:3][:]


# In[40]:


#é interessante utilizar o copy pois se vc alterar o arr2, modificará também o arr
arr2 = arr[:][:4].copy


# In[43]:


arr2


# In[44]:


arr > 30


# # 3. Operações com Numpy Arrays

# In[45]:


arr = np.arange(0,16)
arr


# In[47]:


#Tem que ter o mesmo tamanho
arr+arr


# In[48]:


arr + 30


# In[49]:


#raiz quadrada
np.sqrt(arr)


# In[50]:


np.exp(arr)


# In[51]:


#média
np.mean(arr)


# In[53]:


#desvio padrão, std
np.std(arr)


# In[54]:


np.sin(arr)


# In[56]:


np.max(arr)


# In[ ]:




