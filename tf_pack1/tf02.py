# 변수 : 모델 학습 시, 매개변수 갱신 등을 위해 사용

import tensorflow as tf
f = tf.Variable(1.0)
v = tf.Variable(tf.ones((2,)))
m = tf.Variable(tf.ones((2, 1)))
print(f) # numpy=1.0
print(v) # numpy=array([1., 1.]
print(m)
print(m.numpy())

print()
v1 = tf.Variable(1)  # 0-d tensor
print(v1)
# v1 = 77 -> AttributeError: 'int' object has no attribute 'assign'
v1.assign(10) # 값 치환
print(v1, v1.numpy(), type(v1))
# <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=10> 10 <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>

v2 = tf.Variable(tf.ones(shape=(1)))  # 1-d tensor
v2.assign([20])
print(v2, v2.numpy(), type(v2))

v3 = tf.Variable(tf.ones(shape=(1, 2)))  # 2-d tensor
v3.assign([[20, 30]])
print(v3, v3.numpy(), type(v3))

print()
v1 = tf.Variable([3])
v2 = tf.Variable([5])
v3 = v1*v2 + 10
print(v3) # tf.Tensor([25], shape=(1,), dtype=int32)

print()
var = tf.Variable([1,2,3,4,5], dtype=tf.float32)
result1 = var + 10
print(result1) # tf.Tensor([11. 12. 13. 14. 15.], shape=(5,), dtype=float32)

print()
w = tf.Variable(tf.ones(shape=(1,)))
b = tf.Variable(tf.ones(shape=(1,)))
w.assign([2])
b.assign([3])

def func1(x):
    return w*x + b

out_a1 = func1([[3]])  # 3, [3]
print('out_a1 : ', out_a1) # out_a1 :  tf.Tensor([9.], shape=(1,), dtype=float32)

print()
@tf.function  # auto graph 기능이 적용된 함수 : tf.Graph + tf.Session이 적용, 텐서는 텐서끼리 연산해야 빠르다
def func2(x):
    return w*x + b

print(type(func2))
out_a2 = func2([1,2])
print('out_a2 : ', out_a2) # Function클래스

print()
rand = tf.random.uniform([5], 0 ,1)  # uniform 균등분포 - ([5], min=0 , max=1)
print(rand.numpy())
print()
rand2 = tf.random.normal([5], mean=0 , stddev=1)  # normal 정규분포
print(rand2.numpy())

print()
aa = tf.ones((2,1))
print(aa.numpy())

m = tf.Variable(tf.zeros((2,1)))
print(m.numpy())
m.assign(aa)  # 치환
print(m.numpy())

m.assign_add(aa)  # assign_add() => m += aa 이런 개념이다.  누적
print(m.numpy())

m.assign_sub(aa)  # 빼는거
print(m.numpy())



print('\n\n---TF의 구조 (Graph로 설계된 내용은 Session에 실행)---')
g1 = tf.Graph()

with g1.as_default():   # 특정 자원 처리를 한 후 자동 close()
    c1 = tf.constant(1, name='c_one')
    c1_1 = tf.constant(1, name='c_one_1')
    print(c1)  # Tensor("c_one:0", shape=(), dtype=int32)
    print(type(c1))  # Tensor객체
    
    print(c1.op)  # tf.Operation 객체
    print('-------')
    print(g1.as_graph_def())

print('-----------------')
g2 = tf.Graph()
with g2.as_default():
    v1=tf.Variable(initial_value=1, name='v1')
    print(v1)
    print(type(v1))   # ResourceVariable 객체
    print(v1.op)

print('~~~~~~~~~')
print(g2.as_graph_def())

