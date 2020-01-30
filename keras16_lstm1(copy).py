from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터 
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape)  # (13, 3)
print(y.shape)  # (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)


# 2. 모델구성
model = Sequential()
model.add(LSTM(10, activation = 'relu',input_shape = (3,1), return_sequences = True))   # return_sequences = True : LSTM의 아웃풋을 다음 input layer 차원에 맞춰주는 거(defalut : False)
model.add(LSTM(20, activation = 'relu', return_sequences = True))  
# model.add(LSTM(20, activation = 'relu', return_sequences = True)) 
# model.add(LSTM(20, activation = 'relu', return_sequences = True)) 
# model.add(LSTM(20, activation = 'relu', return_sequences = True)) 
# model.add(LSTM(20, activation = 'relu', return_sequences = True)) 
# model.add(LSTM(20, activation = 'relu', return_sequences = True)) 
# model.add(LSTM(20, activation = 'relu', return_sequences = True)) 
# model.add(LSTM(20, activation = 'relu', return_sequences = True)) 
model.add(LSTM(20, activation = 'relu', return_sequences = False)) 
#return sequence를 많이 넣으면 좋지않다. 그 이유는 시계열 데이터를 붕괴시킴.
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(12))
# model.add(Dense(10))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))
# model.add(Dense(5))


model.add(Dense(5))
model.add(Dense(1))

model.summary()

# # 3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])  

# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor= 'acc', patience= 20, mode = 'max') 
# model.fit(x, y, epochs=1000, batch_size=1, verbose=1, callbacks=[early_stopping])  

# # 4. 평가예측
# loss, mae = model.evaluate(x, y, batch_size=1)
# print(loss, mae) 


# x_input = array([[6.5,7.5,8.5],[50,60,70],[70,80,90],[100,110,120]])  # (3, ) -> (1,3) -> (1,3,1)
# x_input = x_input.reshape(4,3,1)

# y_predict = model.predict(x_input)
# print(y_predict)