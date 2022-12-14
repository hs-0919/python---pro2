# RNN을 이용한 텍스트 생성
# 문맥을 반영하여 다음 단어 예측하기

import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences,  to_categorical

# text = """경마장에 있는 말이 뛰고 있다
# 그의 말이 법이다
# 가는 말이 고와야 오는 말이 곱다"""

text = """윤석열 대통령은 13일 '문재인 케어' 폐기를 사실상 공식화하면서 건강보험 재정개혁의 시급성을 적극 강조했다.
노동개혁·연금개혁·교육개혁 등 3대 개혁과제를 추진하는 윤 대통령은 미래세대를 위한 개혁대상으로 건강보험까지 추가로 제시한 것이다.
이 과정에서 윤 대통령은 문재인 정부가 도입했던 보장성 강화를 주요 내용으로 한 이른바 문재인 케어가 건강보험 재정악화의 주요인임을 
언급하며 책임론을 제기, 향후 건강보험 개혁 과정에서 야당과의 마찰이 예상된다."""

tok = Tokenizer()
tok.fit_on_texts([text])  # 속성타입은 리스트여야 한다.
print(tok.word_index) # {'말이': 1, '경마장에': 2, ..., '곱다': 11}
encoded = tok.texts_to_sequences([text])
print(encoded)

vocab_size = len(tok.word_index) + 1 # Embedding(가능한 토큰 수...)
print(vocab_size)

# 훈련 데이터 만들기 
sequences = list()
for line in text.split('\n'):  # 문장 토큰화
    enco = tok.texts_to_sequences([line])[0]
    # print(enco)
    # 바로 다음 단어를 label로 사용하기 위해 리스트에 벡터 기억
    for i in range(1, len(enco)):
        sequ = enco[:i + 1]
        # print(sequ)
        sequences.append(sequ)

print(sequences) # [[2, 3], [2, 3, 1], [2, 3, 1, 4], ... ]
print('학습 참여 샘플 수 : ', len(sequences))  # 11개
print(max(len(i) for i  in sequences))

# 가장 긴 벡터 길이를 기준으로 각 벡터의 크기를 동일
max_len = max(len(i) for i  in sequences)

psequences = pad_sequences(sequences, maxlen=max_len, padding='pre') # post
print(psequences)

# 각 벡터의 마지막 요소(단어를)를 label로 사용
x = psequences[:, :-1]  # feature
y = psequences[:, -1]   # label
print(x)
print(y) # [ 3  1  4  5  1  7  1  9 10  1 11]
# 모델의 최종분류 활성화 함수는 softmax를 사용하므로 y를 원핫 처리

y= to_categorical(y, num_classes=vocab_size)
print(y[:2])

# model
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_len -1))
model.add(LSTM(32, activation='tanh'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=200, verbose=2)
print('model evaluate : ', model.evaluate(x, y))

# 문자열 생성 함수
def seq_gen_text_func(model, t, current_word, n):
    init_word = current_word  # 처음 들어온 단어도 마지막 함께 출력할 계획
    sentence = ''
    for _ in range(n):
        encoded = t.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=max_len - 1, padding='pre')  # - 1 주는 이유 : 마지막에 label을 빼야해서
        result = np.argmax(model.predict(encoded, verbose=0), axis= -1)
        
        # 예측 단어 찾기
        for word, index in t.word_index.items():
            # print(word, ' : ', index)
            if index == result:  # 예측한 단어와 인덱스에 동일한 단어가 있다면 해당 단어는 예측 단어이므로 break
                break
        
        current_word = current_word + ' ' + word
        sentence = sentence + ' '+ word
        
    sentence = init_word + sentence
    return sentence
        
# print(seq_gen_text_func(model, tok, '경마장', 1))
# print(seq_gen_text_func(model, tok, '그의', 2))
# print(seq_gen_text_func(model, tok, '가는', 3))
# print(seq_gen_text_func(model, tok, '경마장에', 4))
print(seq_gen_text_func(model, tok, '대통령', 5))
print(seq_gen_text_func(model, tok, '폐기', 5))
print(seq_gen_text_func(model, tok, '개혁', 5))
