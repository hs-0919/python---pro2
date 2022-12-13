# 네이버 영화 리뷰 데이터로 word2vec 객체 생성 후 특정 단어에 대한 유사도 확인
# !pip install konlpy
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt


urllib.request.urlretrieve("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ratings.txt", filename='rating.txt')
print(pd.read_table('rating.txt'))
train_data = pd.read_table('rating.txt')
print(train_data[:3])
print(len(train_data)) # 10295
print(train_data.isnull().values.any()) # True - null 존재함

train_data = train_data.dropna(how='any')
print(train_data.isnull().values.any())
print(len(train_data)) # 10293

# 한글 외 문자 제거 - 정규표현식 사용하여
train_data['document'] = train_data['document'].str.replace("[^가-힣 ]", "") # 한글과 공백 제거
print(train_data[:5])

# 불용어 제거 
stopwords =['을', '으로', '은', '는', '들', '와', '게', '해서', '하다', '부터']  # 매우 주관적

# Okt로 토큰 처리 
okt = Okt()
token_data = []
for sent in train_data['document']:
    temp = okt.morphs(sent, stem=True)
    temp = [word for word in temp if not word in stopwords]  # 불용어 제거
    token_data.append(temp)
    
print(token_data)

print('리뷰의 최대 길이 : ', max(len(i) for i in token_data))
print('리뷰의 평균 길이 : ', sum(map(len, token_data))/ len(token_data))

plt.hist([len(s) for s in token_data])
plt.xlabel('length of samples')
plt.xlabel('number of samples')
plt.show()

word_modle = Word2Vec(sentences=token_data, size=100, window=5, min_count=5, sg=0)
print(word_modle.wv.vectors.shape)
print(word_modle.wv.most_similar('주인공'))
print(word_modle.wv.most_similar('영화'))

# 사전훈련된 Word2Vec 객체를 사용할 수 있다.
import gensim

model = gensim.models.Word2Vec.load('C:/work/ko.bin')
result = model.wv.most_similar("프로그램")
print(result)
result = model.wv.most_similar("자바")
print(result)
result = model.wv.most_similar("한국어")
print(result)




