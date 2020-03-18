
# coding: utf-8

# In[145]:


import numpy as np
import pandas as pd
import operator
import numbers


# In[146]:


def normalizacao_min_max(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data


# In[147]:


def distancia_Euclidiana(data1, data2, length):
    '''
    Retorna a distância Euclidiana entre dois vetores de dados.
    Input:
        data1, data2: int or float arrays
        length: int. quantidade de atributos que gostaríamos de calcular a distância.
    Output: float
    '''
    dist = 0
    for x in range(length):
        dist += np.power(data1[x] - data2[x], 2)
    return np.sqrt(dist)


# In[204]:


def obter_vizinhos(teste, treinamento, k):
    '''
    Lista de vizinhos ordenados da menor distância Euclidiana para maior
    Input: conjunto de dados para treinamento
    Output: Lista de vizinhos ordenados pela distância
    '''
    distancias = []
    length = len(teste)-1
    for x in range(len(treinamento)):
        dist = distancia_Euclidiana(teste, treinamento.iloc[x,:], length)
        distancias.append((treinamento.iloc[x,:], dist))
        distancias.sort(key=operator.itemgetter(1))
    vizinhos = []
    for x in range(k):
        vizinhos.append(distancias[x][0])
    return vizinhos


# In[205]:


def obter_resposta(vizinhos, atributo):
    '''
    Retorna valor com maior número de votos dentre os vizinhos mais próximos e um determinado atributo.
    Input:
        vizinhos. lista de vizinhos mais próximos
        classe. rótulo ou índice do atributo
    Output:
        int, float or str. Retorna o valor do atributo com maior número de votos entre os vizinhos mais próximos.
    '''
    votos = {}
    for x in range(len(vizinhos)):
        resposta = vizinhos.iloc[x, atributo]
        if resposta in votos:
            votos[resposta] += 1
        else:
            votos[resposta] = 1
    votos_ordenados = sorted(votos.items(), key=operator.itemgetter(1), reverse=True)
    return votos_ordenados[0][0]


# In[206]:


df = pd.read_excel('newbase.xlsx')
df = df.replace('?', np.NaN)
print(df.info())


# In[207]:


numero_de_variaveis = df.shape[1]
numero_de_observacoes = df.shape[0]
print(df.head())


# In[208]:


# Obtém cada tipo para cada atributo(Numérico ou Categórico)
eh_numerico = [all(isinstance(n, numbers.Number) for n in df.iloc[:, i]) for i, x in enumerate(df)]
eh_todo_numerico = sum(eh_numerico) == len(eh_numerico)
eh_todo_categorico = sum(eh_numerico) == 0
eh_tipo_misturado = not eh_todo_categorico and not eh_todo_numerico
    


# In[209]:


# Separa o conjunto de dados em atributos categóricos e numéricos. Normaliza os numéricos.
if eh_tipo_misturado:
    numero_de_variaveis_numericas = sum(eh_numerico)
    numero_de_variaveis_categoricas = numero_de_variaveis - numero_de_variaveis_numericas
    dados_numericos = df.iloc[:, eh_numerico]
    dados_numericos = normalizacao_min_max(dados_numericos) 
    dados_categoricos = df.iloc[:, [not x for x in eh_numerico]]
    print(dados_numericos.head())


# In[210]:


obter_vizinhos(dados_numericos.iloc[9,:], dados_numericos, 3)


# In[211]:


dados_numericos.iloc[9,:]


# In[196]:


vizinhos = obter_vizinhos(dados_numericos.iloc[9,:], dados_numericos, 3)



# In[213]:


np.mean(vizinhos)


# In[ ]:


print(vizinhos.iloc[0,:])


# In[ ]:


dados_numericos_completos.iloc[9,:]


# In[144]:




