# 시퀀스-투-시퀀스(Sequence-to-Sequence, seq2seq)는 입력된 시퀀스로부터 다른 도메인의 시퀀스를 출력하는 
# 다양한 분야에서 사용되는 모델입니다. 번역에 사용되는 모델입니다. 
# RNN에 기반한 seq2seq 모델에는 문제가 있다.  - 하나의 고정된 크기의 벡터에 모든 정보를 압축하려다 보니 정보 손실이 발생한다.
#                                      - 기울기 소실(Vanishing Gradient) 문제가 여전히 발생한다.

# 영어를 불어로 번역하는 번역기 모델 생성 - 글자 단위 기계 번역기 
import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import urllib3

from keras.utils import to_categorical, pad_sequences

http = urllib3.PoolManager()
url ='http://www.manythings.org/anki/kor-eng.zip'
filename = 'kor-eng.zip'
path = os.getcwd()
zipfilename = os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:       
    shutil.copyfileobj(r, out_file)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

lines = pd.read_csv('kor.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic'] # 'lic' 제거
print('전체 샘플의 개수 :',len(lines))

lines = lines.loc[:, 'src':'tar']
lines = lines[0:60000] # 6만개만 저장
lines.sample(10)

# 문자 집합 구축
src_vocab = set()
for line in lines.src: # 1줄씩 읽음
    for char in line: # 1개의 문자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

# 문자 집합의 크기
src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
print('source 문장의 char 집합 :',src_vocab_size)  # source 문장의 char 집합 : 75
print('target 문장의 char 집합 :',tar_vocab_size)  # target 문장의 char 집합 : 921

# 이 중에서 인덱스를 임의로 부여하여 일부만 출력
src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
print(src_vocab[45:75])
print(tar_vocab[45:75])

# 각 문자에 인덱스를 부여
src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
print(src_to_index)
print(tar_to_index)

# 인덱스가 부여된 문자 집합으로부터 갖고있는 훈련 데이터에 정수 인코딩을 수행합니다.
# 우선 인코더의 입력이 될 영어 문장 샘플에 대해서 정수 인코딩을 수행해보고, 5개의 샘플을 출력해봅시다.

encoder_input = []

# 1개의 문장
for line in lines.src:
    encoded_line = []
    # 각 줄에서 1개의 char
    for char in line:
        # 각 char을 정수로 변환
        encoded_line.append(src_to_index[char])
    encoder_input.append(encoded_line)
print('source 문장의 정수 인코딩 :',encoder_input[:5])


# 디코더의 입력이 될 한국어 데이터에 대해서 정수 인코딩을 수행해보겠습니다.

decoder_input = []
for line in lines.tar:
    encoded_line = []
    for char in line:
        encoded_line.append(tar_to_index[char])
    decoder_input.append(encoded_line)
print('target 문장의 정수 인코딩 :',decoder_input[:5])


# 디코더의 예측값과 비교하기 위한 실제값이 필요합니다. 
# 그래서 이번에는 정수 인코딩 과정에서 <sos>를 제거합니다.
# 즉, 모든 한국어 문장의 맨 앞에 붙어있는 '\t'를 제거하도록 합니다.

decoder_target = []
for line in lines.tar:
    timestep = 0
    encoded_line = []
    for char in line:
        if timestep > 0:
            encoded_line.append(tar_to_index[char])
        timestep = timestep + 1
    decoder_target.append(encoded_line)
print('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])


# 패딩을 위해서 영어 문장과 한국어 문장 각각에 대해서 가장 길이가 긴 샘플의 길이를 확인합니다.
max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
print('source 문장의 최대 길이 :',max_src_len)
print('target 문장의 최대 길이 :',max_tar_len)

encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')
print(encoder_input[0])
# 원-핫 벡터를 사용
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)
print(encoder_input[0])

# seq2seq 기계 번역기 훈련
# seq2seq 모델을 설계하고 교사 강요를 사용하여 훈련시켜보도록 하겠습니다.

from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
import numpy as np

# functional api 모델을 구성
encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True)
# 인코더의 내부 상태를 디코더로 넘겨주어야 하기 때문에 return_state=True로 설정

# encoder_outputs은 여기서는 불필요
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# LSTM은 바닐라 RNN과는 달리 상태가 두 개. 은닉 상태와 셀 상태.
encoder_states = [state_h, state_c] # 요게 Context vectors

# encoder_states를 디코더에 전달하므로서 이 두 가지 상태 모두를 디코더로 전달합니다. 
# 이것이 앞서 배운 컨텍스트 벡터입니다.

decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

# 디코더에게 인코더의 은닉 상태, 셀 상태를 전달.
decoder_outputs, _, _= decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
print(model.summary())

model.fit(x=[encoder_input, decoder_input],
          y=decoder_target, batch_size=256, epochs=20, validation_split=0.2)
# batch_size=256 데이터가 많으면 batch_size를 많이 줘야 램에 무리가 안간다.

# seq2seq 기계 번역기 동작시키기
# 전체적인 번역 동작 단계를 정리하면 아래와 같습니다.
# 1. 번역하고자 하는 입력 문장이 인코더에 들어가서 은닉 상태와 셀 상태를 얻습니다.
# 2. 상태와 <SOS>에 해당하는 \t를 디코더로 보냅니다.
# 3. 디코더가 <EOS>에 해당하는 \n이 나올 때까지 다음 문자를 예측하는 행동을 반복합니다.

# 인코더를 정의
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

# 디코더를 설계
# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용.
# 뒤의 함수 decode_sequence()에 동작을 구현 예정
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태를 버리지 않음.
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())
print(index_to_src)
print(index_to_tar)

def decode_sequence(input_seq):
    # 입력으로부터 인코더의 상태를 얻음
    states_value = encoder_model.predict(input_seq)

    # <SOS>에 해당하는 원-핫 벡터 생성
    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, tar_to_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ""

    # stop_condition이 True가 될 때까지 루프 반복
    while not stop_condition:
        # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 예측 결과를 문자로 변환
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]

        # 현재 시점의 예측 문자를 예측 문장에 추가
        decoded_sentence += sampled_char

        # <eos>에 도달하거나 최대 길이를 넘으면 중단.
        if (sampled_char == '\n' or
            len(decoded_sentence) > max_tar_len):
            stop_condition = True

        # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
        target_seq = np.zeros((1, 1, tar_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.

        # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
        states_value = [h, c]

    return decoded_sentence

for seq_index in [3,50,100,300,1001]: # 입력 문장의 인덱스
    input_seq = encoder_input[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(input_seq)
    print(35 * "-")
    print('입력 문장:', lines.src[seq_index])
    print('정답 문장:', lines.tar[seq_index][2:len(lines.tar[seq_index])-1]) # '\t'와 '\n'을 빼고 출력
    print('번역 문장:', decoded_sentence[1:len(decoded_sentence)-1]) # '\n'을 빼고 출력
