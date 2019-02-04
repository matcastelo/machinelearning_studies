#!/usr/bin/env python
# coding: utf-8

# # Introdução aos Pandas
# 
# Nesta seção do curso, aprenderemos a usar pandas para análise de dados. Você deve enxergar o pandas como uma versão extremamente poderosa do Excel, com muito mais recursos. Nesta seção do curso, você deve passar pelos notebooks nesta ordem:
# 
# * Introdução aos Pandas
# * Series
# * DataFrames
# * Dados ausentes
# * GroupBy
# * Mesclar,Juntar, e Concatenar
# * Operações
# * Entrada e saída de dados

# # Series

# In[2]:


import numpy as np
import pandas as pd


# In[2]:


labels = ['a', 'b', 'c']


# In[5]:


minha_lista = [10, 20, 30]
arr = np.array([10, 20, 30])
d = {'a':10, 'b':30, 'c':20}


# In[7]:


# na series os valores estão associados, SERIES SÃO COMO DICIONÁRIOS!
pd.Series(data = minha_lista, index = labels)


# In[8]:


pd.Series([sum, print, len])


# In[9]:


# a ideia é fazer operações com base no index


# # DataFrame - Criação e Fatiamento

# In[9]:


#DataFrames são a base do pandas
arr = np.random.seed(101)
from numpy.random import randn


# In[13]:


df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())

df


# In[20]:


type(df)


# In[30]:


#somando colunas
df['new'] = df['W'] + df['Z']


# In[31]:


df['new']


# In[32]:


#deletando, utiliza-se drop
# axis = 1 eixo da coluna, inplace = True já substuir na variável df
df.drop('new', axis = 1, inplace = True)


# In[33]:


df


# In[36]:


#utilizar loc() para localização (fazer verificação x,y) 

df.loc['A', 'X']


# In[38]:


#lista de valores
df.loc[['A', 'B'], ['X', 'Y']]


# In[39]:


# o iloc() utiliza o padrão de numpy =  de indices numéricos : 
df.iloc[1:2, :3]


# # DataFrame - Seleção condicional, set_index

# In[41]:


df = pd.DataFrame(randn(5,4),index='A B C D E'.split(),columns='W X Y Z'.split())
df


# In[42]:


df > 0.5


# In[43]:


bol = df > 0.5


# In[44]:


df[bol]


# In[47]:


df[df['W']>0]['Y']


# In[49]:


# o & se comporta como and
df[(df['W']>0) & (df['Y'] > 0.2)]


# In[51]:


#set_index
df.reset_index(inplace = True)
df


# # DataFrame - Índices Multiníveis

# In[52]:


# Níveis de Índice
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]

#zip pega duas listas e transforma em tuples
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)


# In[53]:


hier_index


# In[56]:


df = pd.DataFrame(data = np.random.randn(6,2), index = hier_index, columns = 'A B'.split())


# In[57]:


df


# In[60]:


df.loc['G1'].loc[1]


# In[61]:


df.index.names


# In[62]:


df.index.names = ['Grupo', 'N']


# In[63]:


df


# In[64]:


df.xs('G1') #direto para o nível interno


# # Tratamento de dados ausentes

# In[70]:


df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})
df


# In[71]:


df.dropna()


# In[72]:


df.fillna(df.mean())


# In[73]:


df.fillna('vázio')


# In[74]:


df.fillna(df['A'].mean())


# In[79]:


df['A'].fillna(value = df['A'].mean())


# # GroupBy

# # Concatenar, merge e juntar

# In[10]:


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 


df3 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 


# In[81]:


df1


# In[5]:


df4 = pd.DataFrame({'col1':[3, 4, 5, 5],
                  'col2':[4, 5, 6, 2],
                  'col3':[3, 4, 5, 6]},
                 index = ['L1', 'L2', 'L3', 'L4'])


# In[6]:


df4


# ## concatenação
# unir dois dataframes, tem quer o mesmo tamanho no eixo que está unindo

# In[11]:


pd.concat([df1, df2, df3])


# In[13]:


pd.concat([df1, df2, df3], axis =1)


# In[16]:


arr = [[2, 3, 4, 5,],
       [5, 3, 4, 5]]
linha = 'A B'.split()
col = 'col1 col2 col3 co4'.split()


# In[17]:


df = pd.DataFrame( data = arr, index = linha, columns = col)


# In[18]:


df


# In[22]:


arr_news = np.random.randn(5,5)


# In[23]:


arr_news


# In[25]:


linha = 'L1 L2 L3 L4 L5'.split()
col = 'C1 C2 C3 C4 C5'.split()


# In[26]:


df_new = pd.DataFrame(data = arr_news, index = linha, columns = col)


# In[27]:


df_new


# # Operações com Pandas

# In[28]:


import pandas as pd
df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})
df.head()


# In[29]:


#retornar a série de valores de determinada coluna
df['col2'].unique()


# In[32]:


import numpy as np


# In[33]:


np.unique('col2')


# In[36]:


df['col2'].value_counts()


# In[37]:


df[(df['col1']>2) & (df['col2']== 444)]


# In[38]:


def vezes2(x):
    return x*2


# In[39]:


# o método apply é semelhante ao map que aplica determina funçao para todas variaveis
df['col1'].apply(vezes2)


# In[40]:


df['col1'].apply( lambda x: x*x)


# In[41]:


df.index


# In[42]:


df.columns


# In[43]:


df


# In[49]:


data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)

df


# In[47]:


df.pivot_table(values = ['D'], index = ['A', 'B'], columns = ['C'])


# # Entrada e saída de dados

# In[51]:


import pandas as pd
import numpy as np


# In[53]:


df = pd.read_csv('exemplo')


# In[55]:


df + 1 


# In[56]:


df.to_csv('exemplo csv', sep = ';', decimal = ',')


# In[ ]:




