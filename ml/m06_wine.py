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

#1. 데이터 읽어 들이기
wine = pd.read_csv('./data/winequality-white.csv', sep=';', encoding='utf-8')


y = wine["quality"]
x = wine.drop('quality', axis =1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size= 0.2, train_size = 0.8)

#2. 모델구성
model  = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

aaa = model.score(x_test, y_test)
print("aaa", aaa)

y_pred = model.predict(x_test)
print("정답률", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))