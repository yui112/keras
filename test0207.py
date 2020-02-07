import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, Input
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, TensorBoard 
from sklearn.metrics import mean_squared_error, r2_score
from keras.callbacks import EarlyStopping

#1월까지만 넣은 데이터, 2월안넣어도 잘 나와서 1월까지만 했습니다.
#데이터 불러오기
data = pd.read_csv('samsung.csv', encoding="euc-kr")

#데이터 전처리
data = data.sort_index(ascending=False)
data.columns = ['date', 'start', 'highest', 'lowest', 'end', 'amount']

#null값 유무 체크
data.isnull().sum()

#인덱스 기준잡기
data = data.set_index('date')

#x, y 추출
x_samsung = data[['start', 'highest', 'lowest', 'amount']]
x_samsung
y_samsung = data['end']

#플롯형태로 바꾸기 위해 특수문제 제거
x_samsung['start'] = x_samsung['start'].str.replace(',', '')
x_samsung['highest'] = x_samsung['highest'].str.replace(',', '')
x_samsung['lowest'] = x_samsung['lowest'].str.replace(',', '')
x_samsung['amount'] = x_samsung['amount'].str.replace(',', '')


y_samsung = y_samsung.str.replace(',', '')

y_samsung

#shape 확인과 데이터 형식 변환
print(x_samsung.shape)
print(y_samsung.shape)

x_samsung.astype('float64')
y_samsung.astype('float64')

x_samsung.to_numpy()
y_samsung.to_numpy()

#train, test 분류
xs_train, xs_test, ys_train, ys_test = train_test_split(x_samsung, y_samsung, test_size = 0.2, shuffle=False)

#LSTM 모델 학습
xs_train = np.array(xs_train)
xs_train = xs_train.reshape(xs_train.shape[0], xs_train.shape[1], 1)


xs_test = np.array(xs_test)
xs_test = xs_test.reshape(xs_test.shape[0], xs_test.shape[1], 1)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(4,1), return_sequences = True))
model.add(LSTM(20, activation = 'relu', return_sequences = True))
model.add(LSTM(20, activation = 'relu', return_sequences = False))
model.add(Dense(40))
model.add(Dense(80))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])

early_stopping = EarlyStopping(patience=20)

model.fit(xs_train, ys_train, epochs=1500, batch_size = 5, callbacks=[early_stopping]) 

#결과예측
loss, mse = model.evaluate(xs_test, ys_test,batch_size=1)
print('mse : ', mse)

y_predict = model.predict(xs_test,batch_size=1)

# RMSE 구하기
from sklearn.metrics import mean_squared_error

def RMSE(ys_test,y_predict):
    return np.sqrt(mean_squared_error(ys_test,y_predict))

print("RMSE:",RMSE(ys_test,y_predict))

#2월6일짜 데이터를 활용하여 2월7일 종가 
feb06_samsung = np.array([60100, 61100, 59700, 14727159])

feb06_samsung = feb06_samsung.reshape(1,4,1)

feb07_samsung = model.predict(feb06_samsung)

print(feb07_samsung)

# 2월7일 종가 예측
# mse :  399.8163757324219
# RMSE: 481.2220391103983
# [[60167.508]]