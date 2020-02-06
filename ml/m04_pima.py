from keras.models import Sequential
from keras.layers import Dense
import numpy


#import tensorflow as tf

#data로드

dataset = numpy.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:,8]

#모델의 설정
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#모델 컴파일
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

#모델 실행
model.fit(X, Y, epochs=200, batch_size=10)

print("\n Accuracy; %.4f" % (model.evaluate(X, Y)[1]))