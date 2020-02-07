#m10 그리드서치 + m12 파이프라인 합체
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.utils.testing import all_estimators
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:,'Name']
x = iris_data.loc[:,['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]

# 학습 전용과 테스트 전용 분리하기
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    train_size=0.7, shuffle = True)


parameters = { 
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [3,4,5, 6],
    'criterion' :['gini', 'entropy']
}


pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])


model = GridSearchCV(pipe, param_grid=parameters)

model.fit(x_train, y_train)

print("최적의 매개 변수 =", model.best_estimator_)

y_pred = model.predict(x_test)
print("최종 정답률 =", accuracy_score(y_test, y_pred))

