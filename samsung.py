import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, Input
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, TensorBoard 
from sklearn.metrics import mean_squared_error, r2_score

df1 = pd.read_csv('samsung.csv', index_col=0,
                  header=0, encoding='cp949', sep=',')

print(df1.shape)

df2 = pd.read_csv('kospi200.csv', index_col=0,
                  header=0, encoding='cp949', sep=',')

print(df2.shape)

for i in range(df1.index):
    df1.iloc[i,4] = int(df1.iloc[i,4].replace(',', ''))
    
