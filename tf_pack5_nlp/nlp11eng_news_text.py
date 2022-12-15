# RNN으로 양문 학습 후 영문 글 생성 - 글자 단위 학습

filename = 'news.txt'
et = open(filename, encoding='utf-8').read().lower()
print(et)

# 중복문자 제거 후 글자 인덱싱
chars = sorted(list(set(et)))
print(chars)
# 글자 인덱싱
char_to_int = dict((c, i) for i, c in enumerate(chars))
print(char_to_int)

n_chars = len(et)
n_vocab = len(chars)
print('전체 글자수 : ', n_chars)
print('전체 어휘수 : ', n_vocab)

# dict 데이터로 feature와 label 생성

seq_length = 10  # 학습할 문자를 10개씩 끊어서 처리 
datax = []
datay = []

for i in range(0, n_chars - seq_length, 1): # 1은 증가치
    seq_in = et[i:i + seq_length]
    seq_out = et[i + seq_length]
    # print(seq_in, ':', seq_out)
    datax.append([char_to_int[char] for char in seq_in])
    datay.append(char_to_int[seq_out])

print(datax)
print(datay)

datax_patterns = len(datax)
print("datax의 행렬 유형 수 : ", datax_patterns)

# feature의 구조 변경 / 2차원에서 3차원으로
import numpy as np
from keras.utils import to_categorical
feature = np.reshape(datax, (datax_patterns, seq_length, 1))
print(feature[:2], feature.shape)

# 정규화
feature = feature / float(n_vocab)  
print(feature[:2])

label = to_categorical(datay)
print(label[:2])


# model 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys 
import matplotlib.pyplot as plt

model =Sequential()  # 글자단위의 RNN 처리에서는 Embedding을 사용하지 않는다.
model.add(LSTM(units=256, input_shape=(feature.shape[1], feature.shape[2]), activation='tanh',
               return_sequences=True))
model.add(Dropout(rate=0.2))

model.add(LSTM(units=256, input_shape=(feature.shape[1], feature.shape[2]), activation='tanh'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=label.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

chkpoint = ModelCheckpoint('nlp11.hdf5', monitor='loss', verbose=0, save_best_only=True, 
                           mode='min', patience=3)

es = EarlyStopping(monitor='loss', patience=10)
history = model.fit(feature, label, batch_size =32, epochs=100, verbose=2, callbacks=[chkpoint, es])

# 시각화 
fig , loss_ax= plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], label = 'train loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['accuracy'], label = 'train accuracy', c='b')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='lower left')

plt.show()

# 문장 생성
int_to_char = dict((i,c) for i, c in enumerate(chars))
print('int_to_char, : ', int_to_char)

start = np.random.randint(0, len(datax)-1)
pattern = datax[start] # 문자의 시작 글자 랜덤하게 선택
print('pattern : ', pattern)

print('seed : ', )
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
for i in range(500):  # 글자 500개로 문장 만들기
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    pred = model.predict(x, verbose=0)
    # print(np.argmax(pred))
    index = np.argmax(pred)
    result = int_to_char[index]
    seq_iin = [int_to_char[value] for value in pattern]
    # print(result)
    sys.stdout.write(result)
    pattern.append(index)  # 예측된 글자 누적
    pattern = pattern[1:len(pattern)]






