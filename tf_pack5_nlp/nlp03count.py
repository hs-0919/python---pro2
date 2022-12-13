# sklearn이 제공하는 자연어 특징 추출 : 문자열을 수치 벡터화
# 각 텍스트에서 단어 출현 횟수를 카운팅하는 방법으로 CountVectorizer : 수치 벡터화(BOW)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
content = ['How to format my hard disk', 'How disk format format problems ']

# CountVectorizer 연습
count_vec = CountVectorizer(analyzer='word', min_df=1) # 단어단위 - 'word' / 글자단위 - 'char' / min_df=1 최소빈도수=1
print(count_vec)

tran = count_vec.fit_transform(raw_documents=content) # token 처리 후 벡터화
print(tran)
print(count_vec.get_feature_names_out())
# ['disk' 'format' 'hard' 'how' 'my' 'problems' 'to']<== BOW 벡터
#    0        1      2      3    4        5       6  <== 사전 순으로 인덱싱
print(tran.toarray()) # 의미있는 단어를 알기에 문제가 있다.

print('--------------')
# TfidfVectorizer
# TF : 특정 단어가 하나의 문장 안에서 등장하는 횟수.
# DF : 특정단어가 여러 문장에 등장하는 횟수.
# IDF : DF에 역수를 취함.
# TF-IDF : 하나의 문장 안에서 자주나오는 단어에 대해 가중치를 준다. 여러 문장에서 자주 등장하는 단어의 경우에는 패널티를 주는 방법.

tfidf_vec = TfidfVectorizer(analyzer='word', min_df=1)
tran_idf = tfidf_vec.fit_transform(raw_documents=content) # token 처리 후 벡터화
print(tran_idf)
print(tfidf_vec.get_feature_names_out())
print(tran_idf.toarray())




