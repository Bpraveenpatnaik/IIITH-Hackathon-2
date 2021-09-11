# -*- coding: utf-8 -*-
"""Hackathon_2_housing_price.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q0Zpy2g4C8k2JG7kNuwuCi_614pC8J5g
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x),'display.max_columns', None) #Limiting floats output to 3 decimal points

import warnings
def ignore_warn(*args, **kwargs):
    pass
    warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

df = pd.read_csv('train.csv')

df.head()

df.dtypes

df.describe()

sns.displot(df['SalePrice'])
plt.show()

correlation = df.corr()
fig, axes = plt.subplots(figsize=(15,12))
sns.heatmap(correlation, vmax=8)

k = 10 #number of variables for heatmap
cols = correlation.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.2)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df[cols], size = 2.5)

# pivot table for the missing values
total = df.isnull().sum().sort_values(ascending=False)
percentage = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percentage], axis=1, keys=['Total', 'Percentage'])
missing_data.head(20)

df1 = df.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType'], axis=1)

df1.dtypes

df1.head()

df1 = pd.get_dummies(df1)

df1.head()

df1.dtypes

print(df1.columns.tolist())

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from math import sqrt

scaler = StandardScaler()
X = df1.drop('SalePrice', axis=1)
y = df1[['SalePrice']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

scorer = make_scorer(mean_squared_error, greater_is_better = False)
rmse_train = np.sqrt(-cross_val_score(lr, X_train, y_train, scoring = scorer, cv=10))
rmse_test = np.sqrt(-cross_val_score(lr, X_test, y_test, scoring = scorer, cv=10))
print ('Mean RMSE for training set is',rmse_train.mean())
print ('Mean RMSE for the test set is',rmse_test.mean())

import pickle

pickle.dump(lr, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

print(model)

Pkl_Filename = "Pickle_RL_Model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(lr, file)

with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)

Pickled_LR_Model

from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
fit = pca.fit(X)
print(("Explained Variance: %s") % (fit.explained_variance_ratio_))
print("\n")
#print(fit.components_)

pca.explained_variance_ratio_.cumsum()

pc = pca.fit_transform(X)
pc_df = pd.DataFrame(data = pc , 
        columns = ['PC1', 'PC2','PC3','PC4','PC5'])
pc_df['Cluster'] = y
pc_df.head()

X1 = pc_df.drop('Cluster', axis=1)
y1 = pc_df[['Cluster']]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

print(X1_train.shape)
print(X1_test.shape)
print(y1_train.shape)
print(y1_test.shape)

lr = LinearRegression()
lr.fit(X1_train, y1_train)
y1_train_pred = lr.predict(X1_train)
y1_test_pred = lr.predict(X1_test)

scorer = make_scorer(mean_squared_error, greater_is_better = False)
rmse_train = np.sqrt(-cross_val_score(lr, X1_train, y1_train, scoring = scorer, cv=10))
rmse_test = np.sqrt(-cross_val_score(lr, X1_test, y1_test, scoring = scorer, cv=10))
print ('Mean RMSE for training set is',rmse_train.mean())
print ('Mean RMSE for the test set is',rmse_test.mean())

df1.corr()

pip install pyspark

!apt-get install openjdk-8-jdk-headless -qq > /dev/null

# !wget -q https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop3.2.tgz
!wget -q https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz
# unzip the spark file to the current folder
# !tar xf spark-3.0.0-bin-hadoop3.2.tgz
!tar xf spark-3.1.2-bin-hadoop3.2.tgz

# set your spark folder to your system path environment. 
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
# os.environ["SPARK_HOME"] = "/content/spark-3.0.0-bin-hadoop3.2"
os.environ["SPARK_HOME"] = "/content/spark-3.1.2-bin-hadoop3.2"

# install findspark using pip
!pip install -q findspark

os.getcwd()

import findspark
findspark.init()

SPARK_HOME  = '/content/spark-3.1.2-bin-hadoop3.2'
PATH = '%SPARK_HOME%/bin;%SPARK_HOME%/python;%PATH%'
PYTHONPATH = '%SPARK_HOME%\python;%SPARK_HOME%\python\lib\py4j-0.10.9-src.zip:%PYTHONPATH%'

# Check the pyspark version
import pyspark
print(pyspark.__version__)

print(PYTHONPATH)

print(pyspark.conf)

print(pyspark.sql.session)

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

print(spark)

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

dataset = spark.read.csv('train.csv',inferSchema=True, header =True)

dataset.show(5)

dataset.dtypes

df = dataset['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','ExterQual','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','SalePrice']

df1 = df.describe()

df1.show()

num = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','ExterQual','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','SalePrice']

from pyspark.sql.functions import col
# df.select(*(col(c).cast("float").alias(c) for c in df.columns))
for c in num:
    df = df.withColumn(c, df[c].cast('double'))

df = df.withColumnRenamed('SalePrice','label')

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['MSSubClass',
'LotFrontage',
'LotArea',
'OverallQual',
'OverallCond',
'YearBuilt',
'YearRemodAdd',
'ExterQual',
'BsmtFinSF1',
'BsmtFinSF2',
'BsmtUnfSF',
'TotalBsmtSF',
'1stFlrSF',
'2ndFlrSF',
'LowQualFinSF',
'GrLivArea',
'BsmtFullBath',
'BsmtHalfBath',
'FullBath',
'HalfBath',
'BedroomAbvGr',
'KitchenAbvGr',
'TotRmsAbvGrd',
'Fireplaces',
'GarageYrBlt',
'GarageCars',
'GarageArea',
'WoodDeckSF',
'OpenPorchSF',
'EnclosedPorch',
'3SsnPorch',
'ScreenPorch',
'PoolArea',
'MiscVal',
'MoSold',
'YrSold',
'label'], outputCol = 'features')

df = vectorAssembler.transform(df)
finalized_data = df.select(['features', 'label'])
# df3.show(3)

# Define a random forest model.
rf1 = RandomForestRegressor(
    n_estimators=40, max_depth=None, min_samples_leaf=1, min_samples_split=2,
    max_features=.7, max_samples=None, n_jobs=-1, random_state=1)

# Call the utility function for model fitting & evaluation.
mfe(rf1, x_train_tf, y_train, x_val_tf, y_val)

from pyspark.ml.regression import LinearRegression
#Split training and testing data
train_data,test_data = finalized_data.randomSplit([0.8,0.2])

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
#  Fit the model
lrModel = lr.fit(train_data)

# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))


# #To predict the prices on testing set
pred = lrModel.evaluate(test_data)

# #Predict the model
pred.predictions.show()

