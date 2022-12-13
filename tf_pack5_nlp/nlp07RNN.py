#  Recurrent Neural Networks(RNN)은 히든 노드가 방향을 가진 엣지로 연결돼 순환구조를 이루는(directed cycle) 인공신경망의 한 종류입니다. 
# 음성, 문자 등 순차적으로 등장하는 데이터 처리에 적합한 모델로 알려져 있는데요. 
# Convolutional Neural Networks(CNN)과 더불어 최근 들어 각광 받고 있는 알고리즘입니다.

# RNN의 대상은 sequence data: 입력값에 대해 현재의 state가 다음의 state에 영향을 준다
# 활용: 텍스트 분류, 품사태깅, 문서요약, 문서작성, 기계번역, 이미지캡션....

# 층을 형성해보자
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Dense

model = Sequential()
### SimpleRNN 단순한RNN (vanila라고도함)
# model.add(SimpleRNN(units=3, input_shape=(2,10)))
# model.add(SimpleRNN(3,input_length=2, input_dim=10)) #위와 동일
### LSTM : Long Short-Term Memory units
model.add(LSTM(units=3, input_shape=(2,10)))

print(model.summary())

# one to many
model=Sequential()
model.add(SimpleRNN(units=3, batch_input_shape=(8,2,10))) # batch 8개 추가
print(model.summary())
# many to many, one to many
model=Sequential()
model.add(SimpleRNN(units=3, batch_input_shape=(8,2,10), return_sequences=True)) # 다음 층으로 모든 은닉상태를 전달달
print(model.summary())


# LSTM으로 다음 숫자 예측하기 (원래 이런거하라고 하는건아닌데 Dense가 흘러가는거 보여주려고하는거에요)
# RNN에 Dense 붙이는것만 확인하면 됩니다.
from numpy import array

x=array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[10,20,30],[20,30,40],[30,40,50]])
y=array([4,5,6,7,8,40,50,60])

print(x, x.shape); print(y, y.shape) #(8, 3)  # (8,)

# x = x.reshape((x.shape[0], x.shape[1], 1)) # 8,3,1
# print(x, x.shape) #(8, 3, 1)

# 모델구성
model=Sequential()
model.add(LSTM(10,activation='tanh',input_shape=(3,1)))
# model.add(LSTM(10,activation='tahn',input_shape=(3,1), return_sequences=True)) #many-to-many
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
print(model.summary())

from keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='loss', patience=5, mode='auto')
model.fit(x,y, epochs=1000, batch_size=1, verbose=2, callbacks=[es])

pred=model.predict(x)
print('예측값:', pred.flatten())
print('실제값: ', y)
x_new=array([[3,5,7]])
new_pred = model.predict(x_new)
print('새로운값: ',new_pred)

