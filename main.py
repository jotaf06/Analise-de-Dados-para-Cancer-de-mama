import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

df = pd.read_csv('brca_mirnaseq.csv', sep = ';', header = 0, decimal = ',')
df.shape

ax = sns.countplot(x = 'class', data = df) #Grafico de barras das classes
df["class"].value_counts() #Conta o numero de amostras de cada classe

df["class"].value_counts(normalize = True) #Conta o numero de amostras de cada classe e retorna as proporções em porcentagem

df.describe() #Resumo estatistico dos dados

x = df.drop('class', axis = 1)
y = df['class']

#Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, stratify = y, random_state = 42)

y_train.value_counts(normalize = True)
y_test.value_counts(normalize = True)

#Realiza o treinamento com Logistic Regression
lrc = LogisticRegression(random_state = 42)

cv_list_lr_baseline = cross_val_score(
    lrc,
    X_train,
    y_train,
    cv = 10,
    scoring = 'balanced_accuracy'
    )

mean_cv_lr_baseline = np.mean(cv_list_lr_baseline) #Media
std_cv_lr_baseline = np.std(cv_list_lr_baseline) #Desvio Padrao

#Imprime a performance do treino com Logistic Regression
print(f"Performance (bac): {round(mean_cv_lr_baseline, 4)} +- {round(std_cv_lr_baseline, 4)}")

#Realiza o treinamento com KNN
knn = Pipeline([
    ('mms', MinMaxScaler()),
    ('skb', SelectKBest(score_func= chi2, k = 10)),
    ('knn', KNeighborsClassifier(
        n_neighbors = 5,
        p = 2,
        weights = 'uniform',))
    ])

cv_list_knn_baseline = cross_val_score(
    knn,
    X_train,
    y_train,
    cv = 10,
    scoring = 'balanced_accuracy'
    )

mean_cv_knn_baseline = np.mean(cv_list_knn_baseline) #Media
std_cv_knn_baseline = np.std(cv_list_knn_baseline) #Desvio Padrao

#Imprime a performance do treino com KNN
print(f"Performance (bac): {round(mean_cv_knn_baseline, 4)} +- {round(std_cv_knn_baseline, 4)}")

#Realiza o treinamento com KNN com distancias euclidianas
cv_list_knn_euclid = cross_val_score(
    knn,
    X_train,
    y_train,
    cv = 10,
    scoring = 'balanced_accuracy'
    )

mean_cv_knn_euclid = np.mean(cv_list_knn_euclid) #Media
std_cv_knn_euclid = np.std(cv_list_knn_euclid) #Desvio Padrao

#Imprime a performance do treino com KNN Euclidianas
print(f"Performance (bac): {round(mean_cv_knn_euclid, 4)} +- {round(std_cv_knn_euclid, 4)}")
