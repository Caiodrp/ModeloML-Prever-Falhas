#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#blibiotecas
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

# Caminho completo do arquivo .csv
input_file = input("Digite o caminho completo do arquivo .csv de entrada: ") 

df = pd.read_csv(input_file) # Lendo o arquivo .csv

df_encoded = pd.get_dummies(df, columns=['type']) #Transformando variáveis cat em dummies
    
X = df_encoded.drop(['failure_type','udi','product_id'], axis=1) # variáveis independentes e retirando as não utilizadas
y = df_encoded['failure_type'] # a variável alvo

# Intanciando a árvore
dt_clf = DecisionTreeClassifier(random_state = 777)

# Inicializar o objeto KFold
kf = KFold(n_splits=5)

# Array para armazenar as métricas de avaliação
scores = []

# Loop de treinamento do modelo
for train_index, test_index in kf.split(X):
    X_train_k = X.iloc[train_index,:]
    X_test_k = X.iloc[test_index,:]
    y_train_k = y[train_index]
    y_test_k = y[test_index]
        
    dt_clf.fit(X_train_k, y_train_k)
    y_pred = dt_clf.predict(X_test_k)
    score = dt_clf.score(X_test_k, y_test_k)
    scores.append(score)
    
    # Armazenar as métricas de avaliação médias
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print("Média das métricas de avaliação:", mean_score)
    print("Desvio padrão das métricas de avaliação:", std_score)
    
    # Prever novos dados
    dados_novos = input("Digite o caminho completo do arquivo .csv de novos dados: ") # path do arquivo  que quer prever
    new_data = pd.read_csv(dados_novos) #Carregando
    new_data_encoded = pd.get_dummies(new_data, columns=['type']) #Preparando as variáveis pro sklearn
    
    X_new = new_data_encoded.drop(['failure_type','udi','product_id'], axis=1) # variáveis independentes e retirando as não utilizadas
    
    y_pred_new = dt_clf.predict(X_new)
    df_results = pd.DataFrame({'rowNumbers': [new_data.shape[0]], 'predictedValues': y_pred_new})
    df_results.to_csv('predicted.csv', index=False)

