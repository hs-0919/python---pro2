# 양방향 LSTM과 어텐션 메커니즘(BiLSTM with Attention mechanism)

# RNN에 기반한 seq2seq 모델에는 크게 두 가지 문제가 있습니다.
# 첫째, 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하니까 정보 손실이 발생합니다.
# 둘째, RNN의 고질적인 문제인 기울기 소실(vanishing gradient) 문제가 존재합니다.
# 이를 위한 대안으로 입력 시퀀스가 길어지면 출력 시퀀스의 정확도가 떨어지는 것을 보정해주기 위한 등장한 기법인 어텐션(attention)을 소개합니다.


#  IMDB 리뷰 데이터 전처리하기
from keras.datasets import imdb
from keras.utils import to_categorical
from keras.utils import pad_sequences

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

print('리뷰의 최대 길이 : {}'.format(max(len(l) for l in X_train)))
print('리뷰의 평균 길이 : {}'.format(sum(map(len, X_train))/len(X_train)))

max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)


# 바다나우 어텐션(Bahdanau Attention)
# 여기서 사용할 어텐션은 바다나우 어텐션(Bahdanau attention)입니다. 
import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, values, query): # 단, key와 value는 같음
    # query shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

# 양방향 LSTM + 어텐션 메커니즘(BiLSTM with Attention Mechanism)
from keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from keras import Input, Model
from keras import optimizers
import os


# model
# - 여기서는 케라스의 함수형 API를 사용합니다. 우선 입력층과 임베딩층을 설계합니다.
sequence_input = Input(shape=(max_len,), dtype='int32')
# 10,000개의 단어들을 128차원의 벡터로 임베딩하도록 설계
embedded_sequences = Embedding(vocab_size, 128, input_length=max_len, mask_zero = True)(sequence_input)

# 단, 여기서는 양방향 LSTM을 두 층을 사용하겠습니다. 
# 우선, 첫번째 층입니다. 두번째 층을 위에 쌓을 예정이므로 return_sequences를 True로 해주어야 합니다.
lstm = Bidirectional(LSTM(64, dropout=0.5, return_sequences = True))(embedded_sequences)
# 두번째 층을 설계합니다. 상태를 리턴받아야 하므로 return_state를 True로 해주어야 합니다
lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional \
  (LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)
# 크기(shape)를 출력
print(lstm.shape, forward_h.shape, forward_c.shape, backward_h.shape, backward_c.shape)

# 양방향 LSTM을 사용할 경우에는 순방향 LSTM과 역방향 LSTM 각각 은닉 상태와 셀 상태를 가지므로, 
# 양방향 LSTM의 은닉 상태와 셀 상태를 사용하려면 두 방향의 LSTM의 상태들을 연결(concatenate)해주면 됩니다.
state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
state_c = Concatenate()([forward_c, backward_c]) # 셀 상태

# 어텐션 메커니즘에서는 은닉 상태를 사용합니다. 이를 입력으로 컨텍스트 벡터(context vector)를 얻습니다
attention = BahdanauAttention(64) # 가중치 크기 정의
context_vector, attention_weights = attention(lstm, state_h)

# 컨텍스트 벡터를 밀집층(dense layer)에 통과시키고, 이진 분류이므로 최종 출력층에 1개의 뉴런을 배치하고, 
# 활성화 함수로 시그모이드 함수를 사용합니다.
dense1 = Dense(20, activation="relu")(context_vector)
dropout = Dropout(0.5)(dense1)
output = Dense(1, activation="sigmoid")(dropout)
model = Model(inputs=sequence_input, outputs=output)
print(model.summary())

# 옵티마이저로 아담 옵티마이저 사용하고, 모델을 컴파일합니다.
# 시그모이드 함수를 사용하므로 손실 함수로 binary_crossentropy를 사용하였습니다. 모델을 학습시킨다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs = 3, batch_size = 256, validation_data=(X_test, y_test), verbose=1)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))




