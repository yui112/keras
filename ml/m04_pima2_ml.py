from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#data로드

dataset = numpy.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
x_train = dataset[:550, 0:8]
y_train = dataset[:550,8]

x_test = dataset[550:, 0:8]
y_test = dataset[550:,8]

model = KNeighborsClassifier(n_neighbors=1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
y_predict = model.predict(x_test)
print("acc = ", accuracy_score(y_test, y_predict))
