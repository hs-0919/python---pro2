# 단순선형회귀모델 작성 : 3가지 방법 경험하기

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np 

# 공부시간에 따른 성적결과 
x_data = np.array([1,2,3,4,5], dtype=np.float32)
y_data = np.array([11,32,55,61,80], dtype=np.float32)
print('r = ', np.corrcoef(x_data, y_data)) # 0.986

print('방법1 : Sequential api 사용 : 가장 단순하다, 레이어에 노드를 순서대로 쌓아 올리는 완전 연결층을 구성')
model = Sequential()
# model.add(Dense(units=1, input_dim=1, activation='linear'))
model.add(Dense(units=2, input_dim=1, activation='linear'))
model.add(Dense(units=1, activation='linear'))


print(model.summary())
opti = optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opti, loss='mse', metrics=['mse', 'mae'])
history = model.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metrics = model.evaluate(x=x_data, y=y_data)
print('loss_metrics : ', loss_metrics)

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model.predict(x_data)))
print('실제값 : ', y_data)
print('예측값 : ', model.predict(x_data).flatten())

new_data = [1.5,2.3,5.7]
print('성적예측값 : ', model.predict(new_data).flatten())

"""
# loss(mse)시각화
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

plt.plot(history.history['mse'], label='평균제곱오차')
plt.xlabel('학습횟수')
plt.show()

plt.plot(x_data, model.predict(x_data), 'b', x_data, y_data, 'ko')
plt.show()
"""

print('\n방법2 : functional api 사용 : Sequential보다는 유연한 구조. 입력 데이터로 여러 층을 공유 하거나 다양한 종류의 입출력 가능')
from keras.layers import Input
from keras.models import Model
# 각 층은 일종의 함수처럼 처리함
inputs = Input(shape=(1,))
# outputs =Dense(units=1, activation='linear')(inputs)  # 이전 층을 다음 층 함수의 입력으로 사용하도록 변수에 할당.
output1 = Dense(units=2, activation='linear')(inputs)
output2 = Dense(units=1, activation='linear')(output1)
model2 = Model(inputs, output2)


# 이하 방법은 방법1과 동일하다.
opti = optimizers.Adam(learning_rate=0.1)
model2.compile(optimizer=opti, loss='mse', metrics=['mse', 'mae'])
model2.fit(x_data, y_data, batch_size=1, epochs=100, verbose=0)
loss_metrics = model2.evaluate(x=x_data, y=y_data, verbose=0)
print('loss_metrics2 : ', loss_metrics)
print('설명력2 : ', r2_score(y_data, model2.predict(x_data, verbose=0)))

print('\n방법3 : sub classing 사용 : 동적인 구조가 필요한 경우. 메소드를 통해 분석가의 생각을 프로그래밍화')
# multi-input, multi-output model : 다중 입출력 모델, 데이터 흐름이 순차적이지 않을 경우에도 사용 가능
x_data = np.array([[1],[2],[3],[4],[5]], dtype=np.float32)
y_data = np.array([5, 32, 55, 61, 80], dtype=np.float32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # 생성자 내에서 필요한 layer를 생성한 후 call 메소드에서 수행하려는 연산을 적어줌
        self.d1 = Dense(2, activation='linear')
        self.d2 = Dense(1, activation='linear')
    
    def call(self, x): # call 은 자동 호출
        inputs = self.d1(x)
        return self.d2(inputs)

model3 = MyModel()

# 이하 방법은 방법1과 동일하다.
opti = optimizers.Adam(learning_rate=0.1)
model3.compile(optimizer=opti, loss='mse', metrics=['mse', 'mae'])
model3.fit(x_data, y_data, batch_size=1, epochs=100, verbose=0)
loss_metrics = model3.evaluate(x=x_data, y=y_data, verbose=0)
print('loss_metrics3 : ', loss_metrics)
print('설명력3 : ', r2_score(y_data, model3.predict(x_data, verbose=0)))

print('\n방법3-1 : sub classing 사용 : 동적인 구조가 필요한 경우. 메소드를 통해 분석가의 생각을 프로그래밍화')
from keras.layers import Layer

# Custom Layer 작성 : 케라스의 정의된 레이어 이외의 새로운 연산을 위하 레이어 혹은 편의를 목적으로 여러 레이어를 하나로 묶어 처리할 경우 사용
class Linear(Layer):
    def __init__(self, units=1):
        super(Linear, self).__init__()
        self.units = units
        
    def build(self, input_shape):  # 모델의 가중치, 편향과 관련된 내용을 기술
        self.w = self.add_weight(shape=(input_shape[-1], self.units), # shape=(input_shape[-1]) 이렇게하면 자동을 크기를 찾는다.
                                 initializer= 'random_normal', trainable=True)  
        self.b = self.add_weight(shape=(self.units, ), initializer='zeros', trainable=True)
    
    
    # build가 call을 부름
    def call(self, inputs):  # 정의된 값들을 이용하여 해당 층의 로직을 기술
        # y = wx + b
        return tf.matmul(inputs, self.w) + self.b  # 행렬곱  / 일차방적식 반환

class MLP(Model):
    def __init__(self):
        super(MLP, self).__init__()
        # self.linear1 = Linear(1)  # 한개 짜리
        self.linear1 = Linear(2)    # 두개 짜리
        self.linear2 = Linear(1)
    
    # call이 build를 부름  ->  build가 call을 부름
    def call(self, inputs):
        # return self.linear1(inputs)
        net = self.linear1(inputs)
        return self.linear2(net)

model4 = MLP()

# 이하 방법은 방법1과 동일하다.
opti = optimizers.Adam(learning_rate=0.1)
model4.compile(optimizer=opti, loss='mse', metrics=['mse', 'mae'])
model4.fit(x_data, y_data, batch_size=1, epochs=100, verbose=0)
loss_metrics = model4.evaluate(x=x_data, y=y_data, verbose=0)
print('loss_metrics4 : ', loss_metrics)
print('설명력4 : ', r2_score(y_data, model4.predict(x_data, verbose=0)))
print(model4.summary()) # 설명력이 음수가 나온경우는 버그(에러), 아니면 데이터양이 너무 적은것이다.


