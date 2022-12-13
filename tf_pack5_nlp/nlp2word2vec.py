# 네이버 뉴스 기사를 읽어 형태소 분석 후 word2vec을 이용하여 단어간 유사도 확인

import pandas as pd
from konlpy.tag import Okt

okt = Okt()
with open('navernews.txt', mode='r', encoding='utf-8') as f:
    lines = f.read().split('\n')
print(lines)
print(len(lines))

wordDic = {}  # 명사만 추출해 단어 수 확인 {'카카오': 7 ...
for line in lines:
    datas = okt.pos(line)  # pos : 품사 태깅
    # print(datas) # 명사만 찾기 ('카카오', 'Noun')
    for word in datas:
        if word[1] == 'Noun': 
            if not(word[0] in wordDic):
                wordDic[word[0]] = 1
            wordDic[word[0]] += 1

print(wordDic)

# 단어 건수별 내림차순 
keys = sorted(wordDic.items(), key=lambda x:x[1], reverse=True)
print(keys)

# DataFrame에 담기
wordList =[]
countList = []

for word, count in keys[:20]:
    wordList.append(word)
    countList.append(count)

df = pd.DataFrame()
df['word'] = wordList
df['count'] = countList
print(df.head(5))

# 이 후 pandas의 다양한 기능들을 활용함
print('***'*20)
results = []
with open('navernews.txt', mode='r', encoding='utf-8') as f:
    lines = f.read().split('\n')

    for line in lines:
        datas = okt.pos(line, stem=True)  # 원형 어근으로 출력 한가한 -> 한가하다.
        # print(datas)
        imsi = []
        for word in datas:
            if not word[1] in ['Foreign', 'Number','Josa','Punctuation','Modifier','Suffix','Exclamation', 'Alpha']: 
                if len(word[0]) >= 2:
                    imsi.append(word[0])
        
        imsi2 = (" ".join(imsi)).strip()  # 좌우 공백 자르기
        results.append(imsi2)
print(results) # ['카카오 카카오 개인 프로필 영역 인스타그램 좋다 누르다 공감 스티커 지난 출시 하다 이용자 다양하다 반응 내놓다 있다', ...

fn = 'naver_clean.txt'
with open(fn, mode='w', encoding='utf-8') as fobj:
    fobj.write('\n'.join(results))
print('성공')

print('\n밀집벡터 생성 방법 중 word2vec 사용 ---  단어 간 유사도를 확인')
from gensim.models import word2vec

lineobj = word2vec.LineSentence(fn) # 객체로 나옴 
print(lineobj)  # LineSentence object

model = word2vec.Word2Vec(lineobj, min_count=1, vector_size=100, window=10, sg= 1)
# sg = 0 : CBOW - 주변 단어로 중심 단어를 예측,/ sg = 1 : SkipGram - 중심 단어로 주변 단어를 예측
print(model)

print(model.wv.most_similar(positive=['카카오']))
print(model.wv.most_similar(positive=['개인'], topn=3))
print(model.wv.most_similar(positive=['카카오', '개인'], topn=3))
print(model.wv.most_similar(negative=['도입']))


