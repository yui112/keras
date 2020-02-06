import matplotlib.pyplot as plt
import pandas as pd

#1 데이터 읽어오기
wine = pd.read_csv('./data/winequality-white.csv', sep=';', encoding='utf-8')

#품질 데이터별 그룹
