from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from keras.layers import Dense, LSTM, Dropout

def split_sequence3(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) -1:
            break
        seq_x, seq_y = sequence[i:end_ix, : ], sequence[end_ix-1, : ]
        X.append(seq_x)
        y.append(seq_y)
    
    return array(X), array(y)

in_seq1 = array([10,20,30 ,40,50,60 ,70,80,90,100])
in_seq2 = array([15,25,35,45,55,65,75,85,95,105])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

print(in_seq1.shape)
print(out_seq.shape)

in_seq1 = in_seq1.reshape(len(in_seq1),1)
in_seq2 = in_seq2.reshape(len(in_seq2),1)
out_seq = out_seq.reshape(len(out_seq),1)

print(in_seq1.shape)
print(out_seq.shape)

from numpy import hstack

dataset = hstack((in_seq1, in_seq2, out_seq))
print(dataset)

n_step = 3

x, y = split_sequence3(dataset, n_step)

for i in range(len(x)):
    print(x[i], y[i])
    
print(x.shape)
print(y.shape)

x = x.reshape(7, 9)

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(9,)))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(3))

#3. 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가
loss, mae = model.evaluate(x, y)
print('mae : ', mae)

x_prd = np.array([[[90, 95, 100],[110, 115, 120],[130,135,140]]])
x_prd = x_prd.reshape(1, 9)
aaa = model.predict(x_prd)
print(aaa)