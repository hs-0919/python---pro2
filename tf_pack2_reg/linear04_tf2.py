# 2.0 버전
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np 

x_data = [1.,2.,3.,4.,5.]
y_data = [1.2,2.0,3.0,3.5,5.5]
print('상관계수 : ', np.corrcoef([x_data, y_data]))  # 0.9749

model = Sequential()
model.add(Dense(units=1, input_dim=1, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model.fit(x_data, y_data, batch_size=1, epochs=100, verbose=0)
print(model.evaluate(x_data, y_data))

pred = model.predict(x_data)
print('예측값 : ', pred.flatten())
print('실제값 : ', y_data)

# 결정계수(R2)
from sklearn.metrics import r2_score
print('결정계수(설명력) : ', r2_score(y_data, pred))

# 시각화
import matplotlib.pyplot as plt
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, pred, 'g')
plt.show()

# 새로운 값으로 예측 
new_x = [1.5, 2.5, 3.3]
print('예측결과 : ', model.predict(new_x).flatten())

