import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# configuração para não exibir os warnings
import warnings
warnings.filterwarnings("ignore")

# CARGA DO DATASET

# Informa a URL de importação do dataset
url = "https://raw.githubusercontent.com/Tatianepr/mvp4-datascience/main/winequality-final.csv"

# Lê o arquivo
dataset = pd.read_csv(url, delimiter=';')

# Mostra as primeiras linhas do dataset
dataset.head()

# SEPARACAO EM CONJUNTO DE TREINO E CONJUNTO DE TESTE COM HOLDOUT

test_size = 0.20 # tamanho do conjunto de teste tenha 20% do dataset
seed = 7 # semente aleatória - numero fixo para reproduzir mesmos resultados

# Separação em conjuntos de treino e teste
array = dataset.values # ignora os primeiros valores de rótulo com nomes das colunas
X = array[:,0:12] # todas as linhas, da coluna 0 até a 12
y = array[:,12] # pega todas as linhas e a coluna 12 com o resultado da classificação

# função "train_test_split" retorna 4 conjuntos - entrada de treino e teste, saída de treino e teste
# shuffle - sorteia, mas o resultado é o mesmo por cusa do seed. O stratify disribui pessoas com e sem diabetes na mesma proporção
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=test_size, shuffle=True, random_state=seed, stratify=y) # holdout com estratificação

# Parâmetros e partições da validação cruzada. a acurácia será usada para avaliar as amostras.
scoring = 'accuracy'
num_particoes = 10 # segmenta em 10 conjuntos a base de treino
# separação estradificada (StratifiedKFold), ou seja, proporcional ao numero de pessoas diabéticas da amostra
kfold = StratifiedKFold(n_splits=num_particoes, shuffle=True, random_state=seed) # validação cruzada com estratificação

X_train
df = pd.DataFrame(X_train)
df.head()

X_train
df = pd.DataFrame(y_train)
df.head(10)
print(y_train[:10])

# MODELO E INFERENCIA
# CRIACAO E AVALIACAO DE MODELOS

np.random.seed(7) # definindo uma semente global

# Lista que armazenará os modelos
models = []

# Criando os modelos e adicionando-os na lista de tuplas de modelos. é o nome do modelo e a instanciação do modelo.
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Listas para armazenar os resultados
results = []
names = []

# Avaliação dos modelos - acurácia
for name, model in models:
    # faz uma avaliação cruzada para cada modelo usando variaveis folf e scoring definidas no bloco anterior
    # constroi 10 modelos e calcula a acurácia de cada um
    # aplica os comandos fit e predition e diz se o modelo foi bom com a acurácia
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results) # guarda os resultados
    names.append(name) # guarda o nome dos modelos
    msg = "%s - Média da acurácia nos 10 conjuntos: %f - Desvio Padrão: %f" % (name, cv_results.mean(), cv_results.std())
    print(msg) # o melhor foi o SVM

# Boxplot de comparação dos modelos - linha laranja é a mediana - melhor foi CART
fig = plt.figure(figsize=(15,10))
fig.suptitle('Comparação dos Modelos')
ax = fig.add_subplot(111)
plt.boxplot(results) # plota os dados da variavel results
ax.set_xticklabels(names) # exibe no eixo o nome dos modelos
plt.show()

# CRIAÇÃO E AVALIAÇÃO DE MODELOS: DADOS PADRONIZADOS E NORMALIZADOS

# faz tratamento nos dados de cada coluna para melhorar a curácia do modelo
# normalização = pega todos os dados de cada coluna e agrupa em converte em 0 e 1 (minimo e máximo)
# padronização = usa o desvio padrão para reduzir a variação dos valores de cada coluna
# o método pipeline faz vários fit's e no final um transform

np.random.seed(7) # definindo uma semente global para este bloco

# Listas para armazenar os pipelines e os resultados para todas as visões do dataset
pipelines = []
results = []
names = []


# Criando os elementos do pipeline

# Algoritmos que serão utilizados
knn = ('KNN', KNeighborsClassifier())
cart = ('CART', DecisionTreeClassifier())
naive_bayes = ('NB', GaussianNB())
svm = ('SVM', SVC())

# Transformações que serão utilizadas
standard_scaler = ('StandardScaler', StandardScaler()) # redimencionador para padronização
min_max_scaler = ('MinMaxScaler', MinMaxScaler()) # faz a normalização


# Montando os pipelines

# Dataset original - executa no padrão original
pipelines.append(('KNN-original', Pipeline([knn])))
pipelines.append(('CART-original', Pipeline([cart])))
pipelines.append(('NB-original', Pipeline([naive_bayes])))
pipelines.append(('SVM-original', Pipeline([svm])))

# Dataset Padronizado
pipelines.append(('KNN-padronizado', Pipeline([standard_scaler, knn])))
pipelines.append(('CART-padronizado', Pipeline([standard_scaler, cart])))
pipelines.append(('NB-padronizado', Pipeline([standard_scaler, naive_bayes])))
pipelines.append(('SVM-padronizado', Pipeline([standard_scaler, svm])))

# Dataset Normalizado
pipelines.append(('KNN-normalizado', Pipeline([min_max_scaler, knn])))
pipelines.append(('CART-normalizado', Pipeline([min_max_scaler, cart])))
pipelines.append(('NB-normalizado', Pipeline([min_max_scaler, naive_bayes])))
pipelines.append(('SVM-normalizado', Pipeline([min_max_scaler, svm])))

# Executando os pipelines
for name, model in pipelines:
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: Média da Acurácia: %.3f Desvio Padrão: %.3f" % (name, cv_results.mean(), cv_results.std()) # formatando para 3 casas decimais
    print(msg)

# Boxplot de comparação dos modelos - melhor foi o SVM padronizado
fig = plt.figure(figsize=(25,6))
fig.suptitle('Comparação dos Modelos - Dataset orginal, padronizado e normalizado -  0.776')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names, rotation=90)
plt.show()

# OTIMIZAÇÃO DOS HIPERPARÂMETROS

# Tuning do KNN

np.random.seed(7) # definindo uma semente global para este bloco

pipelines = []

# Definindo os componentes do pipeline
knn = ('KNN', KNeighborsClassifier())
standard_scaler = ('StandardScaler', StandardScaler())
min_max_scaler = ('MinMaxScaler', MinMaxScaler())

pipelines.append(('knn-original', Pipeline(steps=[knn])))
pipelines.append(('knn-padronizado', Pipeline(steps=[standard_scaler, knn])))
pipelines.append(('knn-normalizado', Pipeline(steps=[min_max_scaler, knn])))

# valores padrão da função KNN quando não informo nada:
# n_neighbors= 5 - usa-se número ímpar para não ter problema com empate
# minkowski = minkowski

param_grid = {
    'KNN__n_neighbors': [1,3,5,7,9,11,13,15,17,19,21],
    'KNN__metric': ["euclidean", "manhattan", "minkowski"],
}

# Prepara e executa o GridSearchCV - busca entre todos os parâmetros usados, o melhor - constroi 99 modelos
# o KNN normalizado com manhattan e 1 vizinho foi o melhor com 0,789695, melhor que CART
for name, model in pipelines:
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid.fit(X_train, y_train) # treina o modelo
    # imprime a melhor configuração
    print("Modelo: %s - Melhor: %f usando %s" % (name, grid.best_score_, grid.best_params_))

# FINALIZAÇÃO DO MODELO

# Avaliação do modelo com o conjunto de testes - usou KNN normalizado, manhattan, com 1 vizinho
# Preparação do modelo
scaler = MinMaxScaler().fit(X_train) # ajuste do scaler com o conjunto de treino
rescaledX = scaler.transform(X_train) # aplicação da normalização no conjunto de treino
model = KNeighborsClassifier(metric='manhattan', n_neighbors=1)
model.fit(rescaledX, y_train) # treina o modelo

# Estimativa da acurácia no conjunto de teste - usando dados que não foram usados "dados de testes"
rescaledTestX = scaler.transform(X_test) # aplicação da normalização com os dados do "scaler" aplicado no conjunto de testes acima
predictions = model.predict(rescaledTestX) # faz uma predição com base no modelo
print(accuracy_score(y_test, predictions)) # avalia a acurácia - 79,61%

# Preparação do modelo com TODO o dataset - MODELO FINAL
scaler = MinMaxScaler().fit(X) # ajuste do scaler com TODO o dataset
rescaledX = scaler.transform(X) # aplicação da padronização com TODO o dataset
model = KNeighborsClassifier(metric='manhattan', n_neighbors=1)
model.fit(rescaledX, y)

# SIMULANDO A APLICAÇÃO DO MODELO EM DADOS NÃO VISTOS

# Novos dados (SÃO 4) - não sabemos a classe!
data = {'type':  [1, 2, 2, 1],
        'fixedacidity': [10, 5.8, 6, 6],
        'volatileacidity': [0.80, 0.24, 0.25, 0.54],
        'citricacid': [0, 0.4, 0.5, 0.5],
        'residualsugar': [6, 3.5, 2, 3],
        'chlorides': [0.078, 0.029, 0.044, 0.044],
        'freesulfurdioxide': [5, 5, 15, 10],
        'totalsulfurdioxide': [40, 109, 29, 47],
        'density': [0.9973, 0.9913, 0.9972, 10.9972],
        'densitypH': [3.5, 3.53, 3.3, 3.3],
        'sulphates': [1, 0.43, 0.89, 0.57],
        'alcohol': [13, 13, 9, 9],
        }

atributos = ['type', 'fixedacidity', 'volatileacidity', 'citricacid', 'residualsugar', 'chlorides', 'freesulfurdioxide', 'totalsulfurdioxide', 'density', 'densitypH', 'sulphates', 'alcohol' ]
entrada = pd.DataFrame(data, columns=atributos) # monta um dataframe com os atributos e dados

array_entrada = entrada.values #ignora nome dos atributos
X_entrada = array_entrada[:,0:12].astype(float)

# Padronização nos dados de entrada usando o scaler utilizado em X
# scaler = MinMaxScaler().fit(X_train)
rescaledEntradaX = scaler.transform(X_entrada)
print(rescaledEntradaX) # apresenta dados normalizados

# Predição de classes dos dados de entrada
# Esse é o modelo que precisa ser exportado "model"
saidas = model.predict(rescaledEntradaX) # faz uma predição com base nos dados fornecidos
print(saidas) # imprime o resultdo para os 4 vinhos apresentados

# salvando o modelo em um arquivo
import pickle

pickle_out = open("modelo_vinho_treinado.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

# Salvar o scaler em um arquivo chamado 'scaler.pkl'
scaler_file = open('scaler_vinho.pkl', 'wb') 
pickle.dump(scaler, scaler_file)

# carrega o arquivo criado
pickle_in  = open("modelo_vinho_treinado.pkl", "rb")
modelo_arq = pickle.load(pickle_in)
pickle_in.close()

scaler_in = open('scaler_vinho.pkl', 'rb')
scaler_arq = pickle.load(scaler_in)
scaler_in.close()


# faço uma predição usando mesmos dados dos 4 vinhos, precisa dar o mesmo resultado
rescaledArqX = scaler_arq.transform(X_entrada)
saidas = modelo_arq.predict(rescaledArqX) # faz uma predição com base nos dados fornecidos
print(saidas) # imprime o resultdo para os 4 vinhos  classificados

scalernovo = MinMaxScaler().fit(X)
Padronizar = scaler.transform(entrada)
saidas2 = modelo_arq.predict(Padronizar)
print(saidas2)