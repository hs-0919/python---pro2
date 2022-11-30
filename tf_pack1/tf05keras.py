# Keras 라이브러리(모듈)로 모델 생성 네트워크 구성하기
# Keras의 기본 개념
# - 케라스의 가장 핵심적인 데이터 구조는 "모델" 이다.
# - 케라스에서 제공하는 시퀀스 모델을 이용하여 레이어를 순차적으로 쉽게 쌓을 수 있다. 
# - 케라스는 Sequential에 Dense 레이어(fully-connected layers 완전히 연결된 레이어)를 쌓는 스택 구조를 사용한다.

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
# 활성화함수 - Activation
from keras.optimizers import SGD, RMSprop, Adam

# 논리회로 분류 모델 생성
# 1) 데이터 수집 및 가공
x = np.array([[0,0],[0,1],[1,0],[1,1]])
print(x)
y = np.array([0,1,1,1])  # or [[0],[1],[1],[1]]

# 2) 모델 네트워크 구성
# - 시퀀스 모델을 생성한 뒤 필요한 레이어를 추가하며 구성한다. 
# model = Sequential()
# model.add(Dense(units=1, input_dim=2)) # 1개로 빠져나가고, 2개가 들어온다 
# model.add(Activation('sigmoid'))
# 위에 처럼 써도 되고 아래처럼 써도 된다.
model = Sequential([
    Dense(units=1, input_dim=2),
    Activation('sigmoid') # sigmoid쓴 이유는 분류니까 / 범주면 softmax 
])

# Dense()의 주요 인자를 보자.
# 첫번째 인자 = 출력 뉴런의 수.
# input_dim = 입력 뉴런의 수. (입력의 차원)
# init = 가중치 초기화 방법.
#      - uniform : 균일 분포
#      - normal : 가우시안 분포
# activation = 활성화 함수.       
#      - linear : 디폴트 값으로 별도 활성화 함수 없이 입력 뉴런과 가중치의 계산 결과 그대로 출력. Ex) 선형 회귀
#      - sigmoid : 이진 분류 문제에서 출력층에 주로 사용되는 활성화 함수.
#      - softmax : 셋 이상을 분류하는 다중 클래스 분류 문제에서 출력층에 주로 사용되는 활성화 함수.
#      - relu : 은닉층에 주로 사용되는 활성화 함수.

# 3) 모델 학습 과정 설정
# - 학습하기 전, 학습에 대한 설정을 수행한다. 손실 함수 및 최적화 방법을 정의. compile() 함수를 사용한다.
# model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
# optimizer - 최적화를 위한 프로그램  , 'sgd' = 확률적 경사 하강법 , metrics- 성능지표
# model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])


# 4) 모델 학습시키기 - 머신러닝에서 학습이란 더 나은 표현(출력)을 찾는 자동화 과정이다.
# - 훈련셋을 이용하여 구성한 모델로 학습 시킨다. fit() 함수를 사용한다.
model.fit(x=x, y=y, batch_size=1, epochs=50, verbose=0)

# 5) 모델 평가
# - 준비된 시험셋으로 학습한 모델을 평가한다. eval‎uate() 함수를 사용
loss_metrics = model.evaluate(x=x, y=y, batch_size=1, verbose=0)
print('loss_metrics : ', loss_metrics) # [0.4204499423503876, 0.75] - [loss, accuracy]

# 6) 모델 사용하기 - 예측값 출력
#  - 임의의 입력으로 모델의 출력을 얻는다. predict() 함수를 사용한다.
pred= model.predict(x, batch_size=1, verbose=0)
print('pred : ', pred)
print('pred : ', pred.flatten()) # 차원 축소
pred = (model.predict(x) > 0.5).astype('int32')
print('pred : ', pred)


# 7) 최적의 모델인 경우 저장 -> 저장 후 읽기
model.save('tf5.hdf5') # 확장자 - .hdf5 대용량데이터 일때 씀
del model
 
from keras.models import load_model
model = load_model('tf5.hdf5')
pred = (model.predict(x) > 0.5).astype('int32')
print('pred : ', pred)


