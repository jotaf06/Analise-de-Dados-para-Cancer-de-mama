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
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv('brca_mirnaseq.csv', sep = ';', header = 0, decimal = ',')

x = df.drop('class', axis = 1)
y = df['class']

#Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, stratify = y, random_state = 42)
y_train.value_counts(normalize = True)
y_test.value_counts(normalize = True)

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

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
lr_pca_test = balanced_accuracy_score(y_test, y_pred)

print("Performance: ", round(lr_pca_test, 4))


#Matriz de confusao Classificacoes corretas e incorretas
ConfusionMatrixDisplay.from_estimator(
    lr,
    X_test,
    y_test
)

#Matriz de confusao Classificacoes corretas e incorretas normalizadas
ConfusionMatrixDisplay.from_estimator(
    lr,
    X_train,
    y_train,
    normalize = 'true'
)

plt.show()
