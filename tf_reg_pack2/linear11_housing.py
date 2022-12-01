# 선형회귀용 다층 분류모델 - Sequential, Functional api

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Concatenate
from sklearn.preprocessing import StandardScaler
from keras import Model
import matplotlib.pyplot as plt

housing = fetch_california_housing()
print(housing.keys())
print(housing.data[:2])
print(housing.target[:2])
print(housing.feature_names)
print(housing.target_names)

x_train_all, x_test, y_train_all, y_test = train_test_split(housing.data, housing.target, random_state=12)
print(x_train_all.shape, x_test.shape, y_train_all.shape, y_test.shape) # (15480, 8) (5160, 8) (15480,) (5160,)

# train의 일부를 validation dataset으로 사용할 목적
x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, random_state=12)
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape) # (11610, 8) (3870, 8) (11610,) (3870,)

# 스케일 조정 : 표준화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.fit_transform(x_valid)
x_test = scaler.fit_transform(x_test)
scaler.inverse_transform(x_test)
print(x_train[:2])

# print('Sequential api : 단순한 네트워크 (설계도) 구성')
model = Sequential()
model.add(Dense(units=30, activation='relu', input_shape = x_train.shape[1:]))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = model.fit(x=x_train, y=y_train,
                     epochs = 20, batch_size = 32,
                      validation_data=(x_valid, y_valid), verbose=2)
print('evaluate :', model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=0))

# predict
x_new = x_test[:3]
y_pred = model.predict(x_new)
print('예측값 :', y_pred.ravel())
print('실제값 :', y_test)

#시각화
plt.plot(range(1,21),history.history['mse'], c='b', alpha=0.3, label='mse')
plt.plot(range(1,21),history.history['val_mse'], c='r',alpha=0.3, label='mse')
plt.xlabel('epoch')
plt.show()






print('Functional api 1 : 이전 방법보다 복잡하고 유연한 네트워크(설계로) 구성')
input_ = Input(shape = x_train.shape[1:])
net1 = Dense(units=30, activation='relu')(input_)# inputs
net2 = Dense(units=30, activation='relu')(net1)
concat = Concatenate()([input_, net2]) # 마지막 은닉층의 출려과 입력층을 연결 (concattenate)
output = Dense(units=1)(concat) # 1개의 노드와 activation Func이 없는 출력층을 만들고 Concatenate 층을 사용 
model2 = Model(inputs=[input_], outputs=[output])

model2.compile(optimizer='adam', loss='mse', metrics=['mse'])
history2 = model2.fit(x=x_train, y=y_train,
                     epochs = 20, batch_size = 32,
                      validation_data=(x_valid, y_valid), verbose=2)
print('evaluate :', model2.evaluate(x=x_test, y=y_test, batch_size=32, verbose=0))
# predict
x_new = x_test[:3]
y_pred = model2.predict(x_new)
print('예측값 :', y_pred.ravel())
print('실제값 :', y_test)

#시각화
plt.plot(range(1,21),history2.history['mse'], c='b', alpha=0.3, label='mse')
plt.plot(range(1,21),history2.history['val_mse'], c='r',alpha=0.3, label='mse')
plt.xlabel('epoch')
plt.show()

print('Functional api 2 : 유연한 네트워크(설계도) 구성 - 일부는 짧은 경로, 일부는 긴 경로')
# 여러 개의 입력을 사용
# 예) 5개의 특성(0 ~ 4)은 짧은 경로, 나머지는 긴 경로 사용
input_a = Input(shape = [5], name = 'wide_input')
input_b = Input(shape = [6], name = 'deep_input')
net1 = Dense(units=30, activation='relu')(input_b)
net2 = Dense(units=30, activation='relu')(net1)
concat = Concatenate()([input_a, net2])
output = Dense(units=1, name='output')(concat)

model3 = Model(inputs=[input_a, input_b], outputs=[output])

model3.compile(optimizer='adam', loss='mse', metrics=['mse'])

# fit 처리시 입력값이 복수
x_train_a, x_train_b = x_train[:,:5], x_train[:,2:] # train용
x_valid_a, x_valid_b = x_valid[:,:5], x_valid[:,2:] # train용
x_test_a, x_test_b = x_test[:,:5], x_test[:,2:] # evaluate
x_new_a, x_new_b = x_test_a[:3], x_test_b[3:] # predict용
print(x_new_a, x_new_b)
history3 = model3.fit(x=(x_train_a,x_train_b), y=y_train,
                     epochs = 20, batch_size = 32,
                      validation_data=((x_valid_a, x_valid_b),y_valid), verbose=0)
print('evaluate :', model3.evaluate(x=(x_test_a, x_test_b), y=y_test, batch_size=32, verbose=0))
# predict
x_new = x_test[:5]
y_pred = model3.predict((x_test_a, x_test_b)) # 왜 에러일까? -> (x_new_a, x_new_b) ? 

print('예측값 :', y_pred.ravel())
print('실제값 :', y_test[:3])

#시각화
plt.plot(range(1,21),history3.history['mse'], c='b', alpha=0.3, label='mse')
plt.plot(range(1,21),history3.history['val_mse'], c='r',alpha=0.3, label='mse')
plt.xlabel('epoch')
plt.show()

