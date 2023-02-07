<h1 align="center"> Modelo Para Prever Falhas </h1>

Um modelo de Machine Learning, utilizando o método de árvore de decisão para prever tipo de falhas em máquinas.
Esse script utiliza arquivos .csv, com variáveis predeterminadas de produtos mecânicos que possuem ou não falhas e qual seu tipo falhas. Através das variáveis como temperatura de processo, temperatura do ambiente, torque, desgaste e velocidade de rotação. O modelo é treinado para prever novos dados com as mesmas caracteristicas afim de determinar qual o tipo de falha os novos produtos irão apresentar dado suas variáveis no momento da verificação por sensores.



<h1 align="left"> Instruções: </h1>

O programa é feito utilizando python 3 e sua instalação é nescessária no ambiente em que for implementado. O relatório de análise de dados é no formato notebook e é preciso uma plataforma que seja capaz de abri-lo.

Ele possui dois modelos com métodos diferentes de árvore de decisão com o mesmo objetivo porém por caminhos diferentes, que podem ser utilizados de acordo com a situação.

Também possui um arquivo em forma de notebook com a explicação do problema, análise dos dados, modelo e o código operacional. Pode ser aplicado tanto em forma de notebook quanto em um terminal ou ambiente virtual, gerando um arquivo .csv com duas colunas (rowNumber, predictedValues) que são o número de linhas e o resultado predito dos novos dados de entrada.

Possui os arquivos .csv que foram utilizados para treinar e testar os modelos, assim como o arquivo resultante da previsão original.

Os pacotes nescessários para rodar todo o projeto são: 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



