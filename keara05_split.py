#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(1, 101))

x_train = x[:60]
y_train = y[:60]

x_test = x[60:80]
y_test = y[60:80]

x_val = x[80:101]
y_val = x[80:101]
# print(x.shape)
# print(y.shape)
# print(x_val)

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential()

model.add(Dense(5, input_dim =1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_data=(x_val, y_val))

#4. 평가
loss, mae = model.evaluate(x_test, y_test)
print('mae : ', mae)

x_prd = np.array([101,102,103])
aaa = model.predict(x_prd, batch_size=1)
print(aaa)
