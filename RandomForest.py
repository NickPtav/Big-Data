#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
resultados_exames = pd.read_csv("https://raw.githubusercontent.com/alura-cursos/reducao-dimensionalidade/master/data-set/exames.csv")
resultados_exames.head()


#  Antes de continuar as classificações:
#  Se fossem utilizados os dados na forma que estão, ocorrerá uma mensagem de erro, pois temos valores nulos. 
#  E o RandomForest não aceita estes valores, por isso, vamos fazer uma limpeza nesses dados antes de continuar.

# In[21]:


#Verifica quais são os valores nulos e soma, células nulas retornam o valor 1. 
resultados_exames.isnull().sum()

#Levando em consideração que são 419 valores nulos no "exame_33", e isso corresponde a 73% dos dados
#a melhor opção é deletar esta coluna


# In[28]:


from sklearn.model_selection import train_test_split
from numpy import random
from sklearn.model_selection import train_test_split
from numpy import random

SEED = 123143
random.seed(SEED)

#Para obter apenas os valores dos exames, vamos excluir a coluna id, diagnostico e a coluna 33 do Df "resultados_exames".
valores_exames = resultados_exames.drop(columns=["id", "diagnostico", "exame_33"])
diagnostico = resultados_exames.diagnostico

#Onde "valores_exames" são nossos valores de importância, e "diagnostico" o resultado a ser obtido.
#Portanto, o "valores_exame" será dividido em partes para ser utilizado de treino, e depois para o teste, para X.
#O mesmo para "diagnostico", para Y.
treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames, diagnostico)


# In[29]:


from sklearn.ensemble import RandomForestClassifier

classificador = RandomForestClassifier(n_estimators = 100)
classificador.fit(treino_x, treino_y)
print("Resultado da classificação %.2f%%" %(classificador.score(teste_x, teste_y)*100))


# Para sabermos se o resultado da classificação foi bom podemos usar um resultado 'base'. Para isso fazemos com que o classificador nos dê um resultado de classificação como se todos os valores de diagnóstico fossem iguais. 

# In[31]:


from sklearn.dummy import DummyClassifier

SEED = 123143
random.seed(SEED)

classificador_bobo = DummyClassifier(strategy = "most_frequent")
classificador_bobo.fit(treino_x, treino_y)
print("Resultado da classificação do Dummy %.2f%%" %(classificador_bobo.score(teste_x, teste_y)*100))


# Assim afirmamos que nossa classificação foi relativamente boa, pois ficou bem acima da linha do Dummy Classifier. 
