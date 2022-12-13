# 문자열(corpus - 자연어 데이터 집합) 토큰화 + LSTM으로 감성분류
# 토큰(token) : text를 단어, 문장, 형태소 별로 나눌 수 있는데 이렇게 나뉜 조각들을 토큰이라고 한다.

import numpy as np 
from keras.preprocessing.text import Tokenizer 
from keras.utils import pad_sequences

docs = ['너무 재밌네요', '최고에요', '참 잘 만든 작품입니다', '추천하고 싶어요', '한 번 더 보고싶네요', '글쎄요', '별로네요', '생각보다 너무 지루해요', '연기가 어색 하더군요', '재미없어요']
labels = np.array([1,1,1,1,1,0,0,0,0,0])

token = Tokenizer()
token.fit_on_texts(docs)  # 정수 인코딩
print(token.word_index)   # 각 단어의 대한 인덱싱 확인

x = token.texts_to_sequences(docs)  # 텍스트를 정수 인덱싱하여 리스트로 반환
print(x)

# padding : 서로 다른 길이의 데이터를 가장 신 데이터의 길이와 같게 만듦
padded_x = pad_sequences(x, 5)
print('padded_x : ', padded_x)

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM


word_size = len(token.word_index) + 1  # Embedding에 입력될 단어의 수를 지정. 가능한 토큰 갯수는 단어 인덱스 최댓값 +1 을 준다.
model = Sequential()
model.add(Embedding(word_size, 8, input_length=5))  # (가능 토큰수, 임베딩 차원, 입력수)
model.add(LSTM(32, activation='tanh'))
model.add(Flatten())

model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_x, labels, epochs=20, verbose=2)
print('evaluate : ', model.evaluate(padded_x, labels))

print('predcit : ', np.where(model.predict(padded_x) > 0.5,  1, 0).ravel())
print('lable : ', labels)


