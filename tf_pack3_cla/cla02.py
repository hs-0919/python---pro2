# classification : 주어진 feature에 대해 label로 학습시켜 데이터를 분류
# hΘ(x) = P(y=1|x;Θ) x:feature, Θ: model parameter∅
# hypothesis function의 출력값은 "주어진 feature x라는 값을 가질 때 class 1에 들어갈 확률"이라는 의미
# P(y=0|x;Θ) + P(y=1|x;Θ) = 1

# 로지스틱 회귀 분석 실습 소스) 2.x

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

np.random.seed(0)

x = np.array([[1,2],[2,3],[3,4],[4,3],[3,2],[2,1]])
y = np.array([[0],[0],[0],[1],[1],[1]])

print('Sequential api 사용---------------------------------')
# model = Sequential([
#     Dense(units = 1, input_dim=2),  # input_shape=(2,)
#     Activation('sigmoid')
# ])
model = Sequential()
model.add(Dense(units=1, input_dim=2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=1, verbose=1)
meval = model.evaluate(x, y)
print(meval)  # [0.209698(loss),  1.0(정확도)]

# 새로운 값으로 예측
pred = model.predict(np.array([[1,2],[10,5]]))
print('예측 결과 : ', pred)     # [[0.16490099] [0.9996613 ]]
print('예측 결과 : ', np.squeeze(np.where(pred > 0.5, 1, 0)))  # [0 1]

for i in pred:
    print(1) if i > 0.5 else print(0)
    
print([1 if i > 0.5 else 0 for i in pred])


print('Functional api 사용---------------------------------')
from keras.models import Model
from keras.layers import Input

inputs = Input(shape=(2,))
outputs = Dense(1, activation='sigmoid')(inputs)
model2= Model(inputs, outputs)

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(x, y, epochs=100, batch_size=1, verbose=1)
meval = model2.evaluate(x, y)
print(meval)

# 새로운 값으로 예측
pred = model.predict(np.array([[1,2],[10,5]]))
print('예측 결과 : ', pred)     # [[0.16490099] [0.9996613 ]]
print('예측 결과 : ', np.squeeze(np.where(pred > 0.5, 1, 0)))  # [0 1]
 