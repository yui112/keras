from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from keras import models, layers, initializers, losses, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

boston = load_boston()

x = boston.data
y = boston.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size= 0.2, train_size = 0.8)

# 모델완성하기
model1 = LinearRegression()
model2 = Ridge()
model3 = Lasso()
 
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

aaa = model1.score(x_test, y_test) 
print('aaa :', aaa) 

bbb = model2.score(x_test, y_test) 
print('bbb :', bbb)

ccc = model3.score(x_test, y_test) 
print('ccc :', ccc)