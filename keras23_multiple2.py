from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
from keras.layers import Dense, LSTM, Dropout

def split_sequence2(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
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

x, y = split_sequence2(dataset, n_step)

for i in range(len(x)):
    print(x[i], y[i])
    
print(x.shape)
print(y.shape)

# x = x.reshape(8, 6)

model = Sequential()
model.add(LSTM(10, activation='relu'))
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

# x_prd = np.array([[90,100],[110,120],[130,140]])
# aaa = model.predict(x_prd)
# print(aaa)