# 선형회귀 모형 작성 : 수식 내용
# tensorflow 사용

import tensorflow as tf
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam

x = [1., 2., 3., 4., 5.]
y = [1.2, 2.8, 3.0, 3.5, 6.0]
print(np.corrcoef(x, y))  # r= 0.937
# 인과 관계가 있다 가정하고 회귀분석 작업을 진행


tf.random.set_seed(123)
w = tf.Variable(tf.random.normal((1,)))
b = tf.Variable(tf.random.normal((1,)))
print(w.numpy(), ' ', b.numpy())

# 선형회귀식을 얻기 위해 cost를 줄여가는 작업
opti = SGD()
def train_func(x, y):  # !중요! - 케라스 없이 
    with tf.GradientTape() as tape:
        hypo = tf.add(tf.multiply(w, x), b)  # y = wx + b
        loss = tf.reduce_mean(tf.square(tf.subtract(hypo, y)))
    grad = tape.gradient(loss, [w,b])  # 자동으로 미분을 계산해준다.
    opti.apply_gradients(zip(grad, [w,b]))
    return loss


w_vals = []
cost_vals = []

for i in range(1, 101):  # epochs
    loss_val = train_func(x, y)
    cost_vals.append(loss_val.numpy())
    w_vals.append(w.numpy())
    if i % 10 == 0 :
        print(loss_val)
    
print('cost : ', cost_vals)
print('w : ', w_vals)


# w, cost를 시각화
print()
import matplotlib.pyplot as plt
plt.plot(w_vals, cost_vals, 's--')
plt.xlabel('w')
plt.ylabel('cost')
plt.show()

print('cost가 최대일때 w:', w.numpy())
print('cost가 최소일때 b:', b.numpy()) # yhat= 0.90847826 * x + 0.6487321

# 선형회귀식으로 시각화
y_pred = tf.multiply(x, w) + b 
print('y_pred : ', y_pred)
print('y : ', y)

plt.scatter(x, y, label='real')
plt.plot(x, y_pred, 'b-', label='pred')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print()
# 새 값으로 정량적 예측
new_x = [3.5, 9.7]
new_pred = tf.multiply(new_x, w) + b 
print('예측값 : ', new_pred.numpy())



