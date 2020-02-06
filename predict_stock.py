#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, Input
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, TensorBoard 
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('samsung.csv', encoding="euc-kr")
kospi = pd.read_csv('kospi200.csv', encoding="euc-kr")

data = data.sort_index(ascending=False)
data.columns = ['date', 'start', 'highest', 'lowest', 'end', 'amount']

kospi = kospi.sort_index(ascending=False)
kospi.columns = ['date', 'start', 'highest', 'lowest', 'end', 'amount']

data.isnull().sum()
kospi.isnull().sum()

data = data.set_index('date')
kospi = kospi.set_index('date')

x_samsung = data[['start', 'highest', 'lowest', 'amount']]
x_samsung
y_samsung = data['end']
x_kospi = kospi[['start', 'highest', 'lowest', 'amount']]
y_kospi = kospi['end']

x_samsung['start'] = x_samsung['start'].str.replace(',', '')
x_samsung['highest'] = x_samsung['highest'].str.replace(',', '')
x_samsung['lowest'] = x_samsung['lowest'].str.replace(',', '')
x_samsung['amount'] = x_samsung['amount'].str.replace(',', '')


y_samsung = y_samsung.str.replace(',', '')

x_kospi.dtypes

x_kospi['amount'] = x_kospi['amount'].str.replace(',', '')

y_samsung

print(x_samsung.shape)
print(y_samsung.shape)
print(x_kospi.shape)
print(y_kospi.shape)

x_samsung.astype('float64')
y_samsung.astype('float64')
x_kospi.astype('float64')
x_kospi.astype('float64')

x_samsung.to_numpy()
y_samsung.to_numpy()
x_kospi.to_numpy()
y_kospi.to_numpy()


xs_train, xs_test, ys_train, ys_test = train_test_split(x_samsung, y_samsung, test_size = 0.2, shuffle=False)
xk_train, xk_test, yk_train, yk_test = train_test_split(x_kospi, y_kospi, test_size = 0.2, shuffle=False)

# 1번 문제

# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(20))
# model.add(Dense(40))
# model.add(Dense(60))
# model.add(Dense(120))
# model.add(Dense(60))
# model.add(Dense(40))
# model.add(Dense(20))
# model.add(Dense(1))

# model.compile(loss='mse', optimizer='adam',
#               metrics=['mae'])
# model.fit(xs_train, ys_train, epochs=50, batch_size=1)


# #4. 평가
# loss, mse = model.evaluate(xs_test, ys_test,batch_size=1)
# print('mse : ', mse)

# y_predict = model.predict(xs_test,batch_size=1)

# # RMSE 구하기
# from sklearn.metrics import mean_squared_error

# def RMSE(y_test,y_predict):
#     return np.sqrt(mean_squared_error(ys_test,y_predict))

# print("RMSE:",RMSE(ys_test,y_predict))


# feb_samsung = np.array([57800, 58400, 56400, 19749457])

# feb_samsung = feb_samsung.reshape(1,4)

# y_predict = model.predict(feb_samsung)

# print(y_predict)



# 2번문제
# xs_train = np.array(xs_train)
# xs_train = xs_train.reshape(xs_train.shape[0], xs_train.shape[1], 1)


# xs_test = np.array(xs_test)
# xs_test = xs_test.reshape(xs_test.shape[0], xs_test.shape[1], 1)

# model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(4,1)))
# model.add(Dense(20))
# model.add(Dense(40))
# model.add(Dense(40))
# model.add(Dense(20))
# model.add(Dense(1))

# model.compile(loss='mse', optimizer='adam',
#               metrics=['mae'])
# model.fit(xs_train, ys_train, epochs=50, batch_size=1)


# loss, mse = model.evaluate(xs_test, ys_test,batch_size=1)
# print('mse : ', mse)

# y_predict = model.predict(xs_test,batch_size=1)

# # RMSE 구하기
# from sklearn.metrics import mean_squared_error

# def RMSE(ys_test,y_predict):
#     return np.sqrt(mean_squared_error(ys_test,y_predict))

# print("RMSE:",RMSE(ys_test,y_predict))


# feb_samsung = np.array([57800, 58400, 56400, 19749457])

# feb_samsung = feb_samsung.reshape(1,4,1)

# y_predict = model.predict(feb_samsung)

# print(y_predict)



# 3번문제

# 함수형 모델 1

input1 = Input(shape = (4, ))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

#함수형 모델 2
input2 = Input(shape = (4, ))
dense21 = Dense(7)(input2)
dense22 = Dense(4)(dense21)
output2 = Dense(1)(dense22)

from keras.layers.merge import concatenate
merge1 = concatenate([output1,output2]) # 모델 1,2 엮기

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)

model = Model(inputs = [input1, input2] , outputs = output) # input 여러개면 리스트형식으로 넣어준다.


# 훈련

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit([xs_train,xk_train], ys_train, epochs=5, batch_size = 1) 

# 평가예측
loss, mse = model.evaluate([xs_test,xk_test], ys_test, batch_size = 1)
print('mse:', mse)



y_predict = model.predict([xs_test, xk_test],batch_size=1)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(ys_test, y_predict))
print('RMSE : ', RMSE(ys_test, y_predict)) # 오차값이기 때문에 값이 적을수록 좋음


feb_samsung = np.array([57800, 58400, 56400, 19749457])
feb_kospi = np.array([290.24, 291.47, 284.53, 101455])
feb_samsung = feb_samsung.reshape(1,4)
feb_kospi = feb_kospi.reshape(1,4)

y_predict = model.predict([feb_samsung, feb_kospi])

print(y_predict)




#4번문제

# xs_train = np.array(xs_train)
# xs_train = xs_train.reshape(xs_train.shape[0], xs_train.shape[1], 1)

# xk_train = np.array(xk_train)
# xk_train = xk_train.reshape(xk_train.shape[0], xk_train.shape[1], 1)

# # 모델 1
# input1 = Input(shape = (4,1))
# model1 = LSTM(10, activation='relu')(input1)
# dense1 = Dense(5)(model1)
# dense2 = Dense(2)(dense1)
# dense3 = Dense(3)(dense2)
# output1 = Dense(1)(dense3)

# # 모델 2
# input2 = Input(shape = (4,1))
# model2 = LSTM(6, activation='relu')(input2)
# dense11 = Dense(4)(model2)
# dense21 = Dense(3)(dense11)
# dense31 = Dense(2)(dense21)
# output2 = Dense(1)(dense31)

# # 2번째 방법
# from keras.layers.merge import Add
# merge1 = Add()([output1,output2])

# middle1 = Dense(4)(merge1)
# middle2 = Dense(3)(middle1)

# # 1번째 output
# output_1 = Dense(2)(middle2)
# output_1 = Dense(1)(output_1) # y의 벡터가 1개이므로 1로 설정

# # 2번째 output
# output_2 = Dense(3)(middle2)
# output_2 = Dense(1)(output_2)

# model = Model(inputs = [input1, input2], outputs = [output_1,output_2])
# model.summary()



# # 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
# model.fit([xs_train,xk_train], [ys_train, yk_train], epochs=5, batch_size = 1) 

# # 평가예측
# loss, mse = model.evaluate([xs_test,xk_test], ys_test, batch_size = 1)
# print('mse:', mse)



# y_predict = model.predict([xs_test, xk_test],batch_size=1)

# # RMSE
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(ys_test, y_predict))
# print('RMSE : ', RMSE(ys_test, y_predict)) # 오차값이기 때문에 값이 적을수록 좋음


# feb_samsung = np.array([57800, 58400, 56400, 19749457])
# feb_kospi = np.array([290.24, 291.47, 284.53, 101455])

# feb_samsung = feb_samsung.reshape(1,4,1)
# feb_kospi = feb_kospi.reshape(1,4,1)

# y_predict = model.predict([feb_samsung, feb_kospi])

# print(y_predict)
