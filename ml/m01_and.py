from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x_train = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_train = [0, 0, 0, 1]

#2. 모델
model = LinearSVC()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 : ", y_predict)
# print("acc = ", accuracy_score([0,0,0,1], y_predict))
print("acc = ", accuracy_score(y_train, y_predict))