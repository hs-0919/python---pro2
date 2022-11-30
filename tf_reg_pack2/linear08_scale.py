# 다중선형회귀모델
# scaleing : feature 간 단위의 차이가 클 경우 정규화/표준화 작업이 효과적 - label에는 적용하지 않음
# 표준화 : (요소값 - 평균) / 표준편차
# 정규화 : (요소값 - 최소값) / (최대값 - 최소값)

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler, RobustScaler
# StandardScaler : 표준화. 이상치가 있으면 불균형
# MinMaxScaler : 정규화. 이상치에 민감
# RobustScaler : 이상치의 영향을 최소화함

data = pd.read_csv('../testdata/Advertising.csv')
del data['no']
print(data.head(3))
print(data.corr())

fdata =data[['tv', 'radio','newspaper']]
ldata =data[['sales']]
print(fdata.head(2))
print(ldata.head(2))


# 스케일링 방법1 - 정규화
# scaler = MinMaxScaler(feature_range=(0, 1))
# fedata = scaler.fit_transform(fdata)
# print(fedata)

fedata = minmax_scale(fdata, feature_range=(0, 1), axis=0, copy=True)
# print(fedata)
print(fdata.head(2))
print(fedata[:2])

# ...

# train / test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(fedata, ldata, shuffle=True,
                                                     test_size=0.3, random_state=123)

model = Sequential()
model.add(Dense(20, input_dim=3, activation='linear'))  # hidden에는  activation='relu'도 가능
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
print(model.summary())

# 모델 구조만 시각화
import tensorflow as tf
tf.keras.utils.plot_model(model, 'lin_model.png')

history = model.fit(x=x_train, y=y_train, epochs=100, batch_size=32, verbose=0,
                    validation_split=0.2) 
# validation_split=0.2이란 train을 다시8:2로 쪼갬, 80 학습 20 검증 ,오버피팅 방지목적

# 모델 평가 후 score확인
loss = model.evaluate(x=x_test, y=y_test, batch_size=32, verbose=0)
print('loss : ', loss[0])

# history 값
print(history.history)
print(history.history['loss'])
print(history.history['val_loss'])
print(history.history['mse'])
print(history.history['val_mse'])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


