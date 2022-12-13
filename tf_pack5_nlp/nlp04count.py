# 한글 데이터로 워드 카운트 하기!
# 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
text_data = ['나는 배가 고프다 아니오리미로', '오늘 점심은 뭐 먹지?', '내일 공부 해야겠다.', '점심 먹고 공부 해!야!지!']

count_vec = CountVectorizer(analyzer='word', min_df=1)
# count_vec = CountVectorizer(analyzer='word', min_df=1, ngram_range=(1,1)) - 기본값
# count_vec = CountVectorizer(analyzer='word', min_df=1, ngram_range=(3,3)) # - 3개씩 묶기
# count_vec = CountVectorizer(analyzer='word', min_df=1, max_df=5)
# count_vec = CountVectorizer(stop_words=['점심', '공부']) # 불용어는 제외 

count_vec.fit(raw_documents=text_data)
print(count_vec.get_feature_names_out())
# ['고프다' '공부' '나는' '내일' '먹고' '먹지' '배가' '아니오리미로' '오늘' '점심' '점심은' '해야겠다'] - 사전순
print(count_vec.vocabulary_)

# transform()으로 벡터화 
print([text_data[0]])
sentense = [text_data[0]]
print(count_vec.transform(sentense))
print(count_vec.transform(sentense).toarray()) # [[1 0 1 0 0 0 1 1 0 0 0 0]]

# 형태소 분석 후 워드 카운트 
from konlpy.tag import Okt

okt= Okt()
my_words = []
for i, doc in enumerate(text_data):
    for word in okt.pos(doc, stem=True):
        # print(word)
        if word[1] in ['Noun', 'Verb', 'Adjective']:
            my_words.append(word[0])

print(my_words)

count_vec = CountVectorizer(analyzer='word', min_df=1, ngram_range=(1,1))
count_vec.fit(my_words)
print(count_vec.get_feature_names_out()) # 한글자는 제외
print(count_vec.vocabulary_)
print(count_vec.transform(my_words))
print(count_vec.transform(my_words).toarray())

print('-----------------')
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer(analyzer='word', min_df=1, ngram_range=(1,1))
tfidf_vec.fit(my_words)
print(tfidf_vec.get_feature_names_out()) # 한글자는 제외
print(tfidf_vec.vocabulary_)
print(tfidf_vec.transform(my_words))
print(tfidf_vec.transform(my_words).toarray())


