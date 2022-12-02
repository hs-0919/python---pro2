# 문제1)
# http://www.randomservices.org/random/data/Galton.txt
# data를 이용해 아버지 키로 아들의 키를 예측하는 회귀분석 모델을 작성하시오.
# train / test 분리
# Sequential api와 function api 를 사용해 모델을 만들어 보시오.
# train과 test의 mse를 시각화 하시오
# 새로운 아버지 키에 대한 자료로 아들의 키를 예측하시오.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

df_data = pd.read_csv("Galtons Height Data.csv", usecols=['Father', 'Gender', 'Height'])
print(df_data.head(5), df_data.shape)

son_data = df_data[df_data['Gender'] == 'M'].drop('Gender', axis=1)
print(son_data.head(2))


# 아버지 아들 키 분리
father_x = son_data.Father
son_y = son_data.Height
print('corrcoef:', np.corrcoef(father_x, son_y))  # 0.391

train_x, test_x, train_y, test_y = train_test_split(father_x, son_y, test_size=0.3, random_state=1)
print(train_x.shape, test_x.shape)  # (325,) (140,)


print('Sequential api 사용------------------------------')
model = Sequential()
model.add(Dense(units=5, input_dim=1, activation='linear'))
model.add(Dense(1, activation='linear'))
print(model.summary())

opti = tf.keras.optimizers.Adam(learning_rate=0.001)     # adam 사용
model.compile(optimizer=opti, loss='mse', metrics=['mse'])

# 모델 학습
history = model.fit(x=train_x, y=train_y, epochs=50, batch_size=4, verbose=0)
loss_metrics = model.evaluate(x=test_x, y=test_y, verbose=0)
print('loss metrics: ', loss_metrics)

print('실제값 : ', test_y.head().values)
print('예측값 :', model.predict(test_x).flatten()[:5])

# 새로운 값 예측
new_height = [75, 70, 80]
print('새로운 예측값: ', model.predict(new_height).flatten())

# 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.plot(train_x, model.predict(train_x), 'b', train_x, train_y, 'ko') # train
plt.show()

plt.plot(test_x, model.predict(test_x), 'b', test_x, test_y, 'ko')  # test
plt.show()

plt.plot(history.history['mse'], label='평균제곱오차')
plt.xlabel('학습 횟수')
plt.show()

print('\n--Function api 사용------------------------------')
from keras.layers import Input
from keras.models import Model

inputs = Input(shape=(1,))
output1 = Dense(units=5, activation='linear')(inputs)
outputs = Dense(1, activation='linear')(output1)

model2 = Model(inputs, outputs)
print(model2.summary())

opti = tf.keras.optimizers.Adam(learning_rate=0.001)     
model2.compile(optimizer=opti, loss='mse', metrics=['mse']) 
history = model2.fit(x=train_x, y=train_y, epochs=50, batch_size=4, verbose=0)
loss_metrics2 = model2.evaluate(x=test_x, y=test_y, verbose=0)
print('loss metrics: ', loss_metrics2)
print('실제값 : ', test_y.head().values)
print('예측값 :', model2.predict(test_x).flatten()[:5])
print()

new_data = [75, 70, 80]
print('새로운 예측 키 값: ', model2.predict(new_data).flatten())




