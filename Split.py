#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix

# Caminho completo do arquivo .csv
input_file = input("Digite o caminho completo do arquivo .csv de entrada: ") 

df = pd.read_csv(input_file) # Lendo o arquivo .csv

df_encoded = pd.get_dummies(df, columns=['type']) #Transformando variáveis cat em dummies
    
X = df_encoded.drop(['failure_type','udi','product_id'], axis=1) # variáveis independentes e retirando as não utilizadas
y = df_encoded['failure_type'] # a variável alvo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123) #Separando a base de treino entre treino e teste

# Instanciando a árvore
dt_clf = DecisionTreeClassifier(random_state = 123)

# Obtendo os ccp_alphas da árvore
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Guardando em uma lista todos os ccp_alphas da árvore
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

#Loop de treino e teste para gerar lista de acurácias
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

#Buscando o melhor alpha que não esteja com overfiting
deltas = [train_scores[i] - test_scores[i] for i in range(len(ccp_alphas))]
min_delta = min(deltas)
best_alpha = ccp_alphas[deltas.index(min_delta)]
print("Best alpha: ", best_alpha)

#Plotando um gráfico para analisar a Acurácia com o alphas de treino e teste
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("Acurácia")
ax.set_title("Acurácia x alpha do conjunto de dados de treino e teste")
ax.plot(ccp_alphas, train_scores, marker='o', label="treino",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="teste",
        drawstyle="steps-post")
ax.legend()
plt.show()

#Matriz de confusão
plot_confusion_matrix(clf1_podada, X_test, y_test,xticks_rotation='vertical')

# Prever novos dados
dados_novos = input("Digite o caminho completo do arquivo .csv de novos dados: ") # path do arquivo  que quer prever
new_data = pd.read_csv(dados_novos) #Carregando
new_data_encoded = pd.get_dummies(new_data, columns=['type']) #Preparando as variáveis pro sklearn
    
X_new = new_data_encoded.drop(['failure_type','udi','product_id'], axis=1) # variáveis independentes e retirando as não utilizadas
    
y_pred_new = dt_clf.predict(X_new)
df_results = pd.DataFrame({'rowNumbers': [new_data.shape[0]], 'predictedValues': y_pred_new})
df_results.to_csv('predicted.csv', index=False)

