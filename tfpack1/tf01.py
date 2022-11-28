import tensorflow as tf 
import os

# SSE 및 AVX 등의 경고는 소스를 빌드 하면 없어지지만, 명시적으로 경고 없애기 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("즉시 실행 모드: ", tf.executing_eagerly()) 
print("GPU ", "사용 가능" if tf.test.is_gpu_available() else "사용 불가능")
print(tf.__version__)


# 텐서(tensor)는 배열(array)이나 행렬(matrix)과 매우 유사한 특수 자료구조
# 텐서는 GPU나 다른 하드웨어 가속기에서 실행할 수 있다는 점만 제외하면 NumPy의 ndarray와 유사하다
# tensorflow는 numpy기반이다. 

print()
# 상수 선언
print(1, type(1))
print(tf.constant(1), type(tf.constant(1)))  # scala  : 0-d tensor
print(tf.constant([1]))   # 모양은list지만 -> 벡터다 vector : 1-d tensor 1차원 배열
print(tf.constant([[1]])) # metrix : 2-d tensor 2차원 배열
print(tf.rank(tf.constant(1)), '', tf.rank(tf.constant([1])), '',tf.rank(tf.constant([[1]])))

print()
a = tf.constant([1,2])
b = tf.constant([3,4])
c = a + b
print(c) # tf.Tensor([4 6], shape=(2,), dtype=int32)

c=tf.add(a,b)
print(c) # tf.Tensor([4 6], shape=(2,), dtype=int32) - 위에 결과와 같다.

print()
# d =tf.constant([3])    # Broadcasting
d =tf.constant([[3]])  # Broadcasting
e =c+d
print(e)

print(1+2)  # 3
# 연산영역이 다르다.
# tensorflow는 Graph영역에서 실행됨(안보임, 숨어있다.) - Graph영역은 병렬연산
print(tf.constant([1]) + tf.constant([2])) # tf.Tensor([3], shape=(1,), dtype=int32)

print()
print(7)
print(tf.convert_to_tensor(7)) #  tf.Tensor(7, shape=(), dtype=int32)
# int -> float
print(tf.convert_to_tensor(7, dtype=tf.float32)) # tf.Tensor(7.0, shape=(), dtype=float32)
print(tf.cast(7, dtype=tf.float32))              # tf.Tensor(7.0, shape=(), dtype=float32)
print(tf.constant(7.0))                          # tf.Tensor(7.0, shape=(), dtype=float32)
# 다 같다.

print()
# numpy의 ndarray와 tensor 사이에 type 변환
import numpy as np
arr = np.array([1, 2])
print(arr, type(arr)) # <class 'numpy.ndarray'>
print(arr + 5) # [6 7]

tfarr = tf.add(arr, 5) # tf.add를 쓰면 ndarray가 자동으로 tensor로 변환됨
print(tfarr) # tf.Tensor([6 7], shape=(2,), dtype=int32)
print(tfarr.numpy())    # numpy type으로 강제 형변환
print(np.add(tfarr, 3)) # numpy type으로 자동 형변환
print(list(tfarr.numpy())) # list type





