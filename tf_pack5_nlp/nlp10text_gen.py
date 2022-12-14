# 뉴욕 타임스 뉴스 기사 중 헤드라인을 읽어 텍스트 생성 연습

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/articlesapril.csv")
# print(df.head(2))
# print(df.columns, len(df.columns))
print(df['headline'].head(2))
print(df.headline.values)  # headline만 필요하다.
print(df['headline'].isnull().values.any())  # False - null은 없다.

headline = []
headline.extend(list(df.headline.values))
print(headline[:10])
print(len(headline)) # 1324개

# 'Unknown'(noise) 자료 제거 
headline = [n for n in headline if n != 'Unknown']

print(headline[:10])
print(len(headline)) # 1214개

# ascii코드 문자, 구두점 제거, 단어 소문자화
from string import punctuation

# print("He하이llo 가a나다.".encode("ascii", errors='ignore').decode())
# print(", python. ".strip(punctuation)) # . , 제거
# print(", python. ".strip(punctuation + ' ')) # . , 제거 + 공백 제거

def replace_func(s):
    s = s.encode('utf8').decode("ascii", errors='ignore') # ascii코드 문자
    return ''.join(c for c in s if c not in punctuation).lower() # 구두점 제거, 단어 소문자화

# sss = ['abc123,..굿굿good']
# print([replace_func(x) for x in sss])

text = [replace_func(x) for x in headline]
print(text[:5])
# 'Former N.F.L. Cheerleaders’ Settlement Offer'
# 'former nfl cheerleaders settlement offer'

# vocabulary
tok = Tokenizer()
tok.fit_on_texts(text)
vocab_size = len(tok.word_index) + 1
print('vocab_size : ', vocab_size)  # 3494

sequences = list()
for line in text:
    enc = tok.texts_to_sequences([line])[0] # 각 샘플에 대한 인덱싱 처리 (정수 인코딩)
    for i in range(1, len(enc)):
        se = enc[:i + 1]
        sequences.append(se)
        
print(sequences) # [[99, 269], [99, 269, 371], ...]

print(tok.word_index) # {'the': 1, 'a': 2, 'to': 3, 'of': 4, 'in': 5, ...
print(sequences[:11]) # [[99, 269], [99, 269, 371], [99, 269, 371, 1115], ...

print('dict items : ', tok.word_index.items()) # dict items :  dict_items([('the', 1), ('a', 2),   ==>  {1:'the', 2:'a' ..} 이렇게 만들기 
index_to_word = {}
for key, value in tok.word_index.items():
    print('key : ', key)
    print('value : ', value)
    index_to_word[value] = key

print(index_to_word[1])

max_len = max(len(i) for i in sequences)
print('max_len : ', max_len) # 24

psequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(psequences[:3])

psequences = np.array(psequences)
x = psequences[:, :-1]  # feature
y = psequences[:, -1]   # label
print(x[:3])
print(y[:3])

# y값 원핫처리
y = to_categorical(y, num_classes=vocab_size)
print(y[:1]) # [[0. 0. 0. ... 0. 0. 0.]]

# --------- 전처리 완료 --------------

# model 

model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_len -1))
model.add(LSTM(128, activation='tanh'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
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

print(seq_gen_text_func(model, tok, 'who', 10))
print(seq_gen_text_func(model, tok, 'how', 10))
print(seq_gen_text_func(model, tok, 'with', 10))
print(seq_gen_text_func(model, tok, 'and', 10))




