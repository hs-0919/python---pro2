# 모델의 정확도가 높을수록 비용함수 값은 낮아진다.

import numpy as np
import math

real = [10,9,3,2,11] # y 실제값
pred = [11,5,2,4,3]  # y 예측값(모델에 의해 얻어진 값이라 가정)

cost = 0 

for i in range(5):
    cost += math.pow(pred[i] - real[i], 2)
    print(cost)

print('cost : ', cost / len(pred))

print()
real = [10,9,3,2,11]  # y 실제값
pred = [11,8,4,3,11]  # y 예측값(모델에 의해 얻어진 값이라 가정)

cost = 0 

for i in range(5):
    cost += math.pow(pred[i] - real[i], 2)
    print(cost)

print('cost : ', cost / len(pred))

print('-----'*20)
# 가중치(W, weight)와 비용함수(cost function, loss, 손실, 비용)의 변화 값을 시각화
import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [1,2,3,4,5]
# y = [2,4,6,8,10]

b = 0   # bias

# hypothesis = x * w + b
# cost = tf.reduce_sum(tf.pow(hypothesis - y, 2)) / len(x)
# 비용함수 = 예측값 - 실제값에 제곱을 하고 그 합에 대한 평균

w_val = []
cost_val = []

for i in range(-30, 50):
    feed_w = i * 0.1
    # print(feed_w)
    hypothesis = tf.multiply(feed_w, x) + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    # cost = tf.reduce_sum(tf.pow(hypothesis - y, 2)) / len(x) => 이거와 같은말이다
    cost_val.append(cost)
    w_val.append(feed_w)
    print(str(i)+' '+', cost :' + str(cost.numpy()) + ', weight : '+ str(feed_w))

plt.plot(w_val, cost_val)
plt.xlabel('W')
plt.ylabel('cost')
plt.show()
