# tf.constant() : 텐서(일반적으로 상수)를 직접 기억
# tf.Variable() : 텐서가 저장된 주소를 참조

import tensorflow as tf
import numpy as np

node1 = tf.constant(3, tf.float32)
node2 = tf.constant(4.0)
print(node1)
print(node2)
imsi = tf.add(node1, node2)
print(imsi)

print()
node3 = tf.Variable(3, dtype=tf.float32)
node4 = tf.Variable(4.0)
print(node3)
print(node4)
node4.assign_add(node3)
print(node4)

print()
a = tf.constant(5)
b = tf.constant(10)
c = tf.multiply(a, b)
print(c)
result = tf.cond(a < b, lambda: tf.add(10, c), lambda: tf.square(a))  # 삼항연산자 : lambda: tf.add(10, c)-true / lambda: tf.square(a)-false 
print('result : ', result.numpy()) # result :  60

print('~~~~~ 함수 관련 : autograph 기능 (Graph 객체 환경에서 작업하도록 함)~~~~~~')

v = tf.Variable(1)

@tf.function
def find_next_func():
    v.assign(v + 1)
    if tf.equal(v % 2, 0): # 짝수냐?
        v.assign(v + 10)

find_next_func()
print(v.numpy())
print(type(find_next_func))  # 일반 함수 <class 'function'> or polymorphic_function.Function

print('####'*5)
print('-- func1 --')
def func1(): # 1부터 3까지 증가
    imsi = tf.constant(0)
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su)
        # imsi = imsi + su
        imsi += su   # 3개다 결과는 같다
        
    return imsi

kbs = func1()
print(kbs.numpy(), '', np.array(kbs))  # 3  3


print('-- func2 --')
imsi = tf.constant(0)

@tf.function
def func2(): # 1부터 3까지 증가
    # imsi = tf.constant(0)
    global imsi # 파이썬은 표시해줘야함
    su = 1
    for _ in range(3):
        # imsi = tf.add(imsi, su)
        imsi = imsi + su
        # imsi += su   # 3개다 결과는 같다
        
    return imsi

mbc = func2()
print(mbc.numpy(), '', np.array(mbc))


print('-- func3 : Variable--')
imsi= tf.Variable(0)

@tf.function
def func3():
    # imsi= tf.Variable(0)  # auto graph에서는 tf.Variable()은 함수 밖에서 선언해야 한다!!!!!!
    su = 1 
    
    for _ in range(3):
        # imsi = tf.add(imsi, su)
        # imsi = imsi + su #     auto graph에서 이거 3개다 err발생
        # imsi += su  # 
        imsi.assign_add(su) #  auto graph에서는 assign_add()만 가능하다
        
    return imsi

sbs = func3()
print(sbs.numpy(), '', np.array(sbs))

print('@@ 구구단 출력 @@')

@tf.function
def gugu1(dan):
    su = 0 # su = tf.constant(0)
    for _ in range(9): # 0~8
        su = tf.add(su, 1) # su에 1을 더함
        # print(su.numpy())
        # print('{}*{}={:2}'.format(dan, su, dan*su)) @tf.function 이거 쓰면 -> err
        # @tf.function 이거 쓰면 에러가 발생!! / auto graph에서는 tensor 연산에만 집중
        
        
gugu1(5)

print()

# @tf.function
def gugu2(dan):
    for i in range(1, 10):
        result = tf.multiply(dan, i)   # 원소 곱하기, tf.matmul() : 행렬곱
        print('{}*{}={:2}'.format(dan, i, result))

gugu2(3)




