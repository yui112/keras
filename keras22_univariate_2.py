from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) -1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    
    return array(X), array(y)



dataset = [10,20,30,40,50,60,70,80,90, 100]

n_steps = 3

x, y = split_sequence(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])

# x = x.reshape(x.shape[0], x.shape[1], 1)

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(3,)))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(1))

#3. 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가
loss, mae = model.evaluate(x, y)
print('mae : ', mae)

x_prd = np.array([[90,100,110]])
aaa = model.predict(x_prd)
print(aaa)