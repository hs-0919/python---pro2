# 스팸(햄)과 스펨(메일) 구분 - RNN으로 분류(이항분류)
# 
import pandas as pd


data = pd.read_csv('spam.csv', encoding='latin1')
print(data.head())
print('샘플 수 : ', len(data))  # 샘플 수 :  5572
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
print(data.head(2))
print(data['v1'].unique())
data['v1'] = data['v1'].replace(['ham', 'spam'], [0,1])
print(data.head(5))

print(data.info()) # null 없음
print(data['v2'].nunique())  # 중복된 데이터가 있다 / 유일 값은 5169이다.
data.drop_duplicates(subset=['v2'],inplace=True)
print('샘플 수 : ', len(data))  # 샘플 수 :  5169

print(data['v1'].value_counts())  # 0(햄): 4516 / 1(스펨메일): 653
print(data.groupby('v1').size().reset_index(name='count'))

# feature / label 분리
x_data = data['v2']
y_data = data['v1']
print(x_data[:1])
print(y_data[:1])

# x_data에 대해 Token 처리
from keras.preprocessing.text import Tokenizer 
tok = Tokenizer()
tok.fit_on_texts(x_data)
sequences = tok.texts_to_sequences(x_data)  # 정수 인덱싱
print(sequences[:5])

word_to_index = tok.word_index  # 각 단어에 부여된 인덱스를 확인
print(word_to_index)

vocab_size = len(word_to_index) + 1
print('vocab_size : ', vocab_size)

# pad_sequence
x_data = sequences
print('메일의 최대 길이 : %d'%max(len(i) for i in x_data))  # 189 
print('메일의 평균 길이 : %f'%(sum(map(len, x_data)) / len(x_data)))  # 15.610370

from keras.utils import pad_sequences
max_len = max(len(i) for i in x_data)
print(max_len)
data = pad_sequences(x_data, maxlen=max_len)
print(data[:1])
print('훈련 데이터의 크기(shape) : ', data.shape)  # 훈련 데이터의 크기(shape) :  (5169, 189)

# train / test split
import numpy as np
n_of_train = int(len(sequences) *0.8)
n_of_test = int(len(sequences) - n_of_train)
print('train 수 : ', n_of_train) # train 수 :  4135
print('test 수 : ', n_of_test) # test 수 :  1034

x_train = data[:n_of_train]
y_train = np.array(y_data[:n_of_train])
x_test = data[n_of_train:]
y_test = np.array(y_data[n_of_train:])
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (4135, 189) (4135,) (1034, 189) (1034,)
print(x_train[:3])
print(y_train[:3])

# 스펨 메일 분류기 작성
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.models import Sequential

model = Sequential()
model.add(Embedding(vocab_size, 32))
model.add(LSTM(32, activation='tanh'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=2)
print('evaluate : ', model.evaluate(x_test, y_test))

# 시각화
import matplotlib.pyplot as plt

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'], label='loss')
plt.plot(epochs, history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(epochs, history.history['acc'], label='acc')
plt.plot(epochs, history.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

# 예측 
pred= model.predict(x_test[:20])  # 20개 참여
print('예측값 : ', np.where(pred > 0.5, 1, 0).flatten())
print('실제값 : ', y_test[:20])
# 잘 예측됨!

