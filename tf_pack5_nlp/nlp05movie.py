# naver 제공 영화 5편을 웹스크래핑 해서 평점을 읽어 영화관 유사도 확인
# 
# !pip install konlpy

from bs4 import BeautifulSoup
import requests
from konlpy.tag import Okt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def movie_scarp_func(url):
    result =[]
    for p in range(1, 11):
        # print(url + "&page=" + str(p))
        r = requests.get(url + "&page=" + str(p))
        # print(r)
        soup = BeautifulSoup(r.content, 'lxml', from_encoding='ms949')
        # print(soup)
        title = soup.find_all('td', {'class':'title'})
        # print(title[0].text)
        sub_result = []
        for i in range(len(title)):
            sub_result.append(title[i].text
                              .replace('\n', '')
                              .replace('\r', '')
                              .replace('\t', '')
                              .replace('별점 - 총 10점 중', '')
                              .replace('신고', '')
                              .replace('올빼미', '')
                              .replace('영화', '')
                              .replace('3D', '')
                              .replace('IMAX', '')
                              .replace('그래비티', '')
            )
        result = result + sub_result
    return("".join(result))


owl = movie_scarp_func("https://movie.naver.com/movie/point/af/list.naver?st=mcode&sword=222301&target=after")
gravity = movie_scarp_func("https://movie.naver.com/movie/point/af/list.naver?st=mcode&sword=47370&target=after")

movies = [owl, gravity]
print(movies)

okt = Okt()

# 형태소 분석 : 명사, 형용사 만 얻기 
def word_sep(movies):
    result = []
    for m in movies:
        words = okt.pos(m)
        one_result = []
        for word in words:
            if word[1] in ['None', 'Adjective'] and len(word[0]) >= 2:
                one_result.append(word[0])
        
        result.append(" ".join(one_result))
    return result

word_list = word_sep(movies)
print(word_list)

# word_list 파일로 저장
import pickle

with open('movie.pickle', 'wb') as fw:
    pickle.dump(word_list, fw)

with open('movie.pickle', 'rb') as fr:
    word_list = pickle.load(fr)
    
print(word_list)

# 1. BOW 추출 방법 1 - CountVectorizer
count = CountVectorizer(analyzer='word', min_df=2)
count_vec = count.fit_transform(raw_documents=word_list).toarray()
# print(count_vec)

pd.set_option('display.max_columns', 500)
count_df = pd.DataFrame(count_vec, columns=count.get_feature_names_out())
print(count_df)



# 2. BOW 추출 방법 2 - TfidfVectorizer
tfidf = TfidfVectorizer(analyzer='word', min_df=2)
count_tfidf = tfidf.fit_transform(raw_documents=word_list).toarray()
# print(count_tfidf)

pd.set_option('display.max_columns', 500)
count_df2 = pd.DataFrame(count_tfidf, columns=tfidf.get_feature_names_out(), index=['owl', 'gravity'])
print(count_df2)

# 영화 간 유사도 - 코사인 유사도를 사용 
def cosine_func(doc1, doc2):
    bunja = sum(doc1 * doc2)
    bunmo = (sum(doc1 ** 2) * sum(doc2 ** 2)) ** 0.5
    return bunja / bunmo

res = np.zeros((2,2))

for i in range(2):
    for j in range(2):
        res[i, j] = cosine_func(count_df2.iloc[i].values, count_df2.iloc[j].values)


# print(res)
df = pd.DataFrame(res, index=['owl', 'gravity'], columns = ['owl', 'gravity'])
print(df)

