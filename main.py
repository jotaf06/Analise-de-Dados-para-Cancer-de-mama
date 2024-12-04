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
        n_neighbors = 3,
        p = 2,
        weights = 'uniform'))
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

#Realiza o treinamento com KNN com distancias manhattan
knn = Pipeline([
    ('mms', MinMaxScaler()),
    ('skb', SelectKBest(score_func= chi2, k = 10)),
    ('knn', KNeighborsClassifier(
        n_neighbors = 3,
        p = 1,
        weights = 'uniform'))
    ])

cv_list_knn_manhattan = cross_val_score(
    knn,
    X_train,
    y_train,
    cv = 10,
    scoring = 'balanced_accuracy'
    )

mean_cv_knn_manhattan = np.mean(cv_list_knn_manhattan) #Media
std_cv_knn_manhattan = np.std(cv_list_knn_manhattan) #Desvio Padrao

#Imprime a performance do treino com KNN Manhattan
print(f"Performance (bac): {round(mean_cv_knn_manhattan, 4)} +- {round(std_cv_knn_manhattan, 4)}")

#Realiza o treinamento com Logistic Regression L2
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        penalty = "l2", #penalidade para evitar overfit
        C = 1, #fator de regularizacao do modelo
        fit_intercept = True, #se o intercepto deve ser estimado
        class_weight = "balanced", #pesos para as classes, pois o dataset eh desbalanceado
        random_state = 42))
    ])

cv_list_lr_l2 = cross_val_score(
    lr,
    X_train,
    y_train,
    cv = 10,
    scoring = 'balanced_accuracy'
    )

mean_cv_lr_l2 = np.mean(cv_list_lr_l2) #Media
std_cv_lr_l2 = np.std(cv_list_lr_l2) #Desvio Padrao

#Imprime a performance do treino com Logistic Regression L2
print(f"Performance (bac): {round(mean_cv_lr_l2, 4)} +- {round(std_cv_lr_l2, 4)}")

#Realiza o treinamento com Logistic Regression L1
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        penalty = "l1", #penalidade para evitar overfit
        C = 1, #fator de regularizacao do modelo
        fit_intercept = True, #se o intercepto deve ser estimado
        class_weight = "balanced", #pesos para as classes, pois o dataset eh desbalanceado
        solver = "liblinear",
        random_state = 42))
    ])

cv_list_lr_l1 = cross_val_score(
    lr,
    X_train,
    y_train,
    cv = 10,
    scoring = 'balanced_accuracy'
    )

mean_cv_lr_l1 = np.mean(cv_list_lr_l1) #Media
std_cv_lr_l1 = np.std(cv_list_lr_l1) #Desvio Padrao

#Imprime a performance do treino com Logistic Regression L2
print(f"Performance (bac): {round(mean_cv_lr_l1, 4)} +- {round(std_cv_lr_l1, 4)}")


lr = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components = 10)),
    ('lr', LogisticRegression(
        penalty = "l2", #penalidade para evitar overfit
        C = 1, #fator de regularizacao do modelo
        fit_intercept = True, #se o intercepto deve ser estimado
        class_weight = "balanced", #pesos para as classes, pois o dataset eh desbalanceado
        solver = "liblinear", #algoritmo de otimizacao usado no treinamento do modelo
        random_state = 42))
])

cv_list_lr_pca = cross_val_score(
    lr,
    X_train,
    y_train,
    cv = 10,
    scoring = 'balanced_accuracy'
    )

mean_cv_lr_pca = np.mean(cv_list_lr_pca) #Media
std_cv_lr_pca = np.std(cv_list_lr_pca) #Desvio Padrao

#Imprime a performance do treino com Logistic Regression L2
print(f"Performance (bac): {round(mean_cv_lr_pca, 4)} +- {round(std_cv_lr_pca, 4)}")

#resultados da cross validation
df_result_cv = pd.DataFrame(
    [cv_list_knn_baseline, cv_list_knn_euclid, cv_list_knn_manhattan, cv_list_lr_l2, cv_list_lr_l1, cv_list_lr_pca],
    index = ['Baseline', 'Knn Euclid', 'Knn Manhattan', 'Logistic Regression L2', 'Logistic Regression L1', 'Logistic Regression PCA'],
).T

df_res = df_result_cv.stack().to_frame("balanced_accuracy")
df_res.index.rename(["fold", "pipelines"], inplace = True) 
df_res = df_res.reset_index()
df_res.head(12)

plt.figure(figsize = (10, 10))
ax = sns.boxplot(x = "pipelines", y = "balanced_accuracy", data = df_res)
ax = sns.swarmplot(x = "pipelines", y = "balanced_accuracy", data = df_res, color = ".40")

#plt.show()
