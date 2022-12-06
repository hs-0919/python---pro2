# diabetes 데이터로 이항분류(sigmoid)와 다항분류(softmax) 처리

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

dataset = np.loadtxt("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/diabetes.csv", delimiter=',')
print(dataset[:1])
print(dataset.shape) # (759, 9)
# print(dataset[:, -1])

# 이항분류 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dataset[:, 0:8], dataset[:, -1], test_size=0.3, random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (531, 8) (228, 8) (531,) (228,)

model =Sequential()
model.add(Dense(units=64, input_dim=8, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0) # epochs=5 편의상 5로 줄임
scores = model.evaluate(x_test, y_test)
print('%s : %.2f'%(model.metrics_names[0], scores[0])) # loss : 0.48     
print('%s : %.2f'%(model.metrics_names[1], scores[1])) # accuracy : 0.80 

# 예측값 구하기
# print(x_train[0])
new_data = [[-0.0588235, 0.20603, 0., 0., 0., -0.105812, -0.910333, -0.433333 ]]
pred = model.predict(new_data, batch_size=32, verbose=0)
print('예측 결과1 : ', pred) # [[0.6487806]] <-- 예측 결과1의 경우는 0.5를 기준으로 0과1로 분류
print('예측 결과1 : ', np.where(pred > 0.5, 1, 0)) 


print('-----'*4)
# 다항 분류는 label을 one hot 인코딩 후 학습에 참여
print(y_train[:3]) # [0. 1. 1.]

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[:3])
# [[1. 0.] / 0 : [1. 0.] , 1: [0. 1.]  바뀜
# [0. 1.]
# [0. 1.]]

model2 =Sequential()
model2.add(Dense(units=64, input_dim=8, activation='relu'))
model2.add(Dense(units=32, activation='relu'))
model2.add(Dense(units=2, activation='softmax'))  # label의 카테고리 수 만큼 결과는 확률값으로 출력
# units=2  -> 빠져나온 값이 두개이므로 유닛은 2이다. 0 : [1. 0.] , 1: [0. 1.]
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0) # epochs=5 편의상 5로 줄임
scores2 = model2.evaluate(x_test, y_test)
print('%s : %.2f'%(model2.metrics_names[0], scores2[0])) #     
print('%s : %.2f'%(model2.metrics_names[1], scores2[1]))

# 예측값 구하기
# print(x_train[0])
new_data2 = [[-0.0588235, 0.20603, 0., 0., 0., -0.105812, -0.910333, -0.433333 ]]
pred2 = model2.predict(new_data2, batch_size=32, verbose=0)
print('예측 결과2 : ', pred2) 
# 예측 결과2 :  [[0.39211503 0.607885  ]]  <-- 예측 결과2의 경우는 확률값이 가장 큰 지점의 인덱스를 분류 결과로 취함
print('예측 결과2 : ', np.argmax(pred2))





