from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[6,7,8]])
y = array([4,5,6,7,8])

x = x.reshape(x.shape[0], x.shape[1], 1)

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1)))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse']) # mse, mae 사용
model.fit(x, y, 
          epochs=100, batch_size = 1) 

# 평가예측
loss, mae = model.evaluate(x,y, batch_size = 1)
print('mse:', loss)
print('lose:', mae)

x_input = array([6,7,8])

x_input = x_input.reshape(1, 3, 1)

y_predict = model.predict(x_input)

print(y_predict)



# print(x.shape)
# print(y.shape)

'''
compile
fit
evaluate
predict
'''