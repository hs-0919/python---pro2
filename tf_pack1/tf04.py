# 연산자와 기본함수 경험
import tensorflow as tf
import numpy as np

x = tf.constant(7)
y = tf.constant(3)

print(x + y)
print(tf.add(x, y))

print(tf.cond(x > y, lambda:tf.add(x,y), lambda:tf.subtract(x,y)))

f1= lambda:tf.constant(123)
f2= lambda:tf.constant(456)

print(tf.case([(tf.greater(x, y), f1)], default=f2))  # tf.less   if(x>y)면 return 123 else return 456

print('관계 연산')
print(tf.equal(1,2).numpy())
print(tf.not_equal(1,2).numpy())
print(tf.greater(1,2).numpy())
print(tf.greater_equal(1,2).numpy())
print(tf.less(1,2).numpy())

print()
print('논리 연산')
print(tf.logical_and(True, False).numpy())
print(tf.logical_or(True, False).numpy())
print(tf.logical_not(True).numpy())

print()
kbs = tf.constant([1,2,2,2,3])
val, idx = tf.unique(kbs)
print(val.numpy())  # value 값
print(idx.numpy())  # index 값

print()
# tf.reduce~ : 차원 축소
ar = [[1,2],[3,4]]
print(tf.reduce_mean(ar).numpy())  # 전체 평균
print(tf.reduce_mean(ar, axis=0).numpy())  # 열 평균
print(tf.reduce_mean(ar, axis=1).numpy())  # 행 평균

t = np.array([[[0,1,2,],[3,4,5]],[[6,7,8],[9,10,11]]])
print(t.shape)
print(tf.reshape(t, shape=[2,6]))   # 차원 변경
print(tf.reshape(t, shape=[-1,6]))  # 차원 변경 - -1쓰면 자동으로 됨(모르면 -1써도 된다.)
print(tf.reshape(t, shape=[2,-1]))  # 차원 변경

print()
# 차원 축소
aa = np.array([[1],[2],[3],[4]])  # 4행 1열이
print(aa.shape)
bb = tf.squeeze(aa)  # 4열로  : squeeze -> 열 요소가 1일때만 차원축소된다.
print(bb.shape)
aa2 = np.array([[[1],[2]],[[3],[4]]])
print(aa2.shape)
print(aa2)
bb2 = tf.squeeze(aa2)
print(bb2.shape)
print(bb2)

print()
print(t.shape)
t2 = tf.squeeze(t)  # 열 요소가 한 개일 때만 차원 축소
print(t2.shape)
print(t2)

print()
# 차원 확대
tarr = tf.constant([[1,2,3],[4,5,6]])
print(tf.shape(tarr))  # 2행 3열 [2 3]
sbs = tf.expand_dims(tarr, 0)  # 첫번째 차원을 추가해 확장 
print(sbs, tf.shape(sbs).numpy())  # 2차원에서 3차원으로
print()
sbs2 = tf.expand_dims(tarr, 1)  # 두번째 차원을 추가해 확장 
print(sbs2, tf.shape(sbs2).numpy())
print()
sbs3 = tf.expand_dims(tarr, 2)  # 세번째 차원을 추가해 확장 
print(sbs3, tf.shape(sbs3).numpy()) # 2면 3행 1열

print()
sbs4 = tf.expand_dims(tarr, -1)  # -1쓰면 뒤에 붙는다.
print(sbs4, tf.shape(sbs4).numpy())

print('~~~~ one-hot ~~~~~')
print(tf.one_hot([0,1,2,3], depth=4))   # depth=4 열갯수 - 갯수 마추기
print(tf.argmax(tf.one_hot([0,1,2,3], depth=4)).numpy())  # [0 1 2 3]


