# 논리게이트 중 XOR는 복수의 뉴런(노드)를 사용

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import SGD, RMSprop, Adam
# 활성화함수 - Activation


# 논리회로 분류 모델 생성
x = np.array([[0,0],[0,1],[1,0],[1,1]])
print(x)
y = np.array([0,1,1,0])  # xor

model = Sequential()

"""
model.add(Dense(units=5, input_dim=2)) # 맨 첫번째만 input_dim을 쓴다.
model.add(Activation('relu')) # 1 layer
model.add(Dense(units=1)) # units=5 5개가 들어와서 units=1 1개로 빠져나감
model.add(Activation('sigmoid')) # 2 layer
"""

# model.add(Flatten(input_shape=(2, )))
# model.add(Dense(units=5, activation='relu'))
model.add(Dense(units=5, input_dim=2, activation='relu'))  # 위 두 주을 한줄로 기술
# model.add(Dense(units=5, input_shape=(2, ), activation='relu')) # 위와 같음
model.add(Dense(units=5, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

print(model.summary()) # 설계된 모델의 layer, parameter 확인

model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy']) 

history = model.fit(x,y, epochs=100, batch_size=1, verbose=0) # history: 학습 도중 발생하는 loss와 acc값을 확인할 수 있다.
print('history loss :  ', history.history['loss'])
print('history acc :  ', history.history['accuracy'])

loss_metrics = model.evaluate(x, y)
print('loss : ', loss_metrics[0], 'acc : ', loss_metrics[1])

pred = (model.predict(x)  > 0.5).astype('int32')
print('예측 결과 : ', pred.flatten())  # 예측 결과 :  [0 1 1 0]

print()
print(model.input)
print(model.output)
print(model.weights)  # kernel(가중치), bias 값 확인

# history 값 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['accuracy'], label='train acc')
plt.xlabel('epochs')
plt.show()

