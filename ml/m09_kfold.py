import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
import numpy as np

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:,'Name']
x = iris_data.loc[:,['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]

Kfold_cv = KFold(n_splits=5, shuffle=True) 


# Classifier 알고리즘 모두 추출하기
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter='classifier')


for(name, algorithm) in allAlgorithms:
    # 각 알고리즘 객체 생성하기
    model = algorithm()
    
    if hasattr(model, "score"): #score만 사용가능한 모델만 불러오는거
        score = cross_validate(model, x, y, cv=Kfold_cv)
        print(name, "의 정답률 =")
        print(score)
        
