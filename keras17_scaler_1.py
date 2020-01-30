from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20000,30000,40000],[30000,40000,50000],
            [40000,50000,60000],[100, 200, 300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler


# scaler = StandardScaler()
# scaler.fit(x)
# x1 = scaler.transform(x)

# print(x1)

# scaler = MinMaxScaler()
# scaler.fit(x)
# x2 = scaler.transform(x)

# print(x2)

# scaler = RobustScaler()
# scaler.fit(x)
# x3 = scaler.transform(x)

#두가지를 합쳐서 할 수도 있다.

x_train = x[:10]
x_test = x[10:]

y_train = y[:10]
y_test = y[10:]



scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x)

# train은 10개, 나머지는 test
# Dense 모델로 구현


from keras.models import Sequential
from keras.layers import Dense, Dropout


# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = Sequential()


model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(1))

model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size = 1)

result = model.evaluate(x_test, y_test, batch_size=1)

print(result)

y_predict = model.predict(x_test,batch_size=1)

from sklearn.metrics import r2_score

r2_y_predict = r2_score(y_test,y_predict)
print("r2:",r2_y_predict)