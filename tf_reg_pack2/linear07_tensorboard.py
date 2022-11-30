# 다중선형회귀모델 작성 후 텐서보드(모델의 구조 및 학습과정/결과를 시각화)

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# 5명의 학생이 3회 시험 실시 후 다음 번 시험 점수 예측
x_data = np.array([[70,85,80], [71,89,78],[50,80,60],[66,20,60],[50,30,10]])
y_data = np.array([73,82,72,57,34])

print('--- Sequential api ---')
model = Sequential()
# model.add(Dense(1, input_dim=3, activation='linear'))  # layer 1개
model.add(Dense(6, input_dim=3, activation='linear', name='a'))  # layer 3개
model.add(Dense(3, activation='linear', name='b'))  
model.add(Dense(1, activation='linear', name='c'))  
print(model.summary())

# 일반적으로 층의 뉴런(노드) 수를 늘리기보다 층수를 늘리는 것이 이득이 많다. - 절대적이지는 않다

opti = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opti, loss='mse', metrics=['mse'])
history = model.fit(x=x_data, y=y_data, batch_size=1, epochs=30, verbose=0)

# plt.plot(history.history['loss'])
# plt.xlabel('epochs')
# plt.xlabel('loss')
# plt.show()

loss_metrics = model.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss_metrics : ', loss_metrics)

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model.predict(x=x_data, batch_size=1, verbose=0)))

print()
print('--- functional api ---')
from keras.layers import Input
from keras.models import Model

inputs = Input(shape=(3, ))
output1 = Dense(6, activation='linear', name='a')(inputs)
output2 = Dense(3, activation='linear', name='b')(output1)
output3 = Dense(1, activation='linear', name='c')(output2)

model2 = Model(inputs, output3)

opti = tf.keras.optimizers.Adam(learning_rate=0.01)
model2.compile(optimizer=opti, loss='mse', metrics=['mse'])

# 텐서보드(TensorBoard) : 알고리즘을 시각화. 복잡한 모델의 수행 도중 발생하는 논리적 오류 등을 개선하기 위한 도구  
from keras.callbacks import TensorBoard

tb = TensorBoard(log_dir='./my',
                 histogram_freq=1,
                 write_graph=True,
                 write_images=False,
                 update_freq='epoch',
                 profile_batch=2,
                 embeddings_freq=1)

history = model2.fit(x=x_data, y=y_data, batch_size=1, epochs=30, verbose=0, callbacks=[tb])
# Tensorboard 실행
# 해당 .py파일이 있는 디렉토리에서 실행
# tensorboard --logdir=my/

loss_metrics = model2.evaluate(x=x_data, y=y_data, batch_size=1, verbose=0)
print('loss_metrics : ', loss_metrics)

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model2.predict(x=x_data, batch_size=1, verbose=0)))


