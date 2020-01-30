#1. 데이터

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
 
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x = x.reshape(x.shape[0], x.shape[1], 1)

#2. 모델구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용

from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='acc', patience=20, mode='max')
#accuracy는 맥스값으로 설정해줌
early_stopping = EarlyStopping(monitor='loss', patience=40, mode='auto')
model.fit(x, y, epochs=1000, batch_size = 1, verbose=1, callbacks=[early_stopping]) 

# 평가예측
loss, mae = model.evaluate(x,y, batch_size = 1)
print('mse:', loss)
print('lose:', mae)

x_input = array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]])

x_input = x_input.reshape(4, 3, 1)

y_predict = model.predict(x_input)

print(y_predict)