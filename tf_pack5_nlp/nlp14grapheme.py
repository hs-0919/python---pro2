# 자소 단위로 한글 데이터를 분리한 후 텍스트 생성 모델
# !pip install jamotools

# 낮은 버전의 파이썬에서 임의 모듈 설치 !pip --use-deprecated=legacy-resolver install  pororo
# https://uiandwe.tistory.com/1315 참고 사이트

import jamotools 
import tensorflow as tf 
import numpy as np 

# short 토지 소설 데이터
path_to_file = tf.keras.utils.get_file("", "https://raw.githubusercontent.com/pykwon/etc/master/rnn_short_toji.txt")
train_text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
s = train_text[:100]  # 100글자 뽑아내기   귀녀의 모습을 ...
print(s)

s_split = jamotools.split_syllables(s)  # ㄱㅟㄴㅕㅇㅢ ㅁㅗㅅㅡㅂㅇㅡㄹ 이런 형태로 분리된다. / 한자는 안잘림
print(s_split)

# 결합
s2 = jamotools.join_jamos(s_split)
print(s2)

# train_text로 자모단위 분리 
train_text_x = jamotools.split_syllables(train_text)
vocab = sorted(set(train_text_x))
vocab.append("UNK")  # "UNK" - 사전에 정의 되지 않은 기호가 있는 경우 'UNK'로 사전에 등록(unknown)
print(vocab)
print(len(vocab))  # 136개

char2idx = {u:i for i, u in enumerate(vocab)} # dict type
print(char2idx)
idx2char = np.array(vocab)
# print(idx2char)

text_as_int = np.array([char2idx[c] for c in train_text_x])
print(text_as_int)  # [35 80 38 ... 10  2  0]
print(train_text_x[:20]) # ㄱㅟㄴㅕㅇㅢ ㅁㅗㅅㅡㅂㅇㅡㄹ ㅎㅏㄴㅂ
print(text_as_int[:20])  # [35 80 38 70 56 83  2 50 72 54 82 51 56 82 43  2 63 64 38 51]

# 학습 데이터 생성
seq_length = 80 
exam_per_epoch = len(text_as_int)  // seq_length
print(exam_per_epoch)  # 하나의 에폭당 8636개처리

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) # 조각을 만들어냄

char_dataset = char_dataset.batch(seq_length + 1, drop_remainder=True)  # 묶어서 처리 -  80개의 자모 ,자소를 / drop_remainder=True -> 나머지는 버림
# 처음 80개의 자소와 그 뒤에 나올 정답이 될 1단어를 합쳐서 반환

for item in char_dataset.take(1):
    print(idx2char[item.numpy()])
    print(item.numpy())

def split_input_target2(chunk): # 끊어서 처리
    return [chunk[:-1], chunk[-1]] # 마지막꺼는 사용하지 않음 레이블로 사용하기위해

train_dataset = char_dataset.map(split_input_target2)

for x, y in train_dataset.take(1):
    print(idx2char[x.numpy()])
    print(x.numpy())
    print(idx2char[y.numpy()])
    print(y.numpy())
    
# model 
batch_size = 64
steps_per_epoch = exam_per_epoch // batch_size
print('steps_per_epoch : ' , steps_per_epoch)
train_dataset = train_dataset.shuffle(buffer_size=5000).batch(batch_size, drop_remainder=True)

total_chars = len(vocab)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_chars, 100, input_length=seq_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=400, activation='tanh'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=total_chars, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())


from keras.utils import pad_sequences

def testmodel(epoch, logs):
    if epoch % 5 != 0 and epoch !=49:
        return
    
    # epoch이 5의 배수 또는 49면 작업을 진행 
    test_sentence = train_text[:48]
    test_sentence = jamotools.split_syllables(test_sentence)
    next_chars = 300
    
    for _ in range(next_chars):
        test_text_x = test_sentence[-seq_length:]
        test_text_x = np.array([char2idx[c] if c in char2idx else char2idx['UNK'] for c in test_text_x])
        test_text_x = pad_sequences([test_text_x], maxlen=seq_length, padding='pre', value=char2idx['UNK'])
        output_idx = np.argmax(model.predict(test_text_x), axis = -1)
        test_sentence += idx2char[output_idx[0]]
        
    print("------")
    print(jamotools.join_jamos(test_sentence))
    
# 모델을 학습시키며 모델이 생성한 결과물을 확인하기 위한 용도
testmodelcb = tf.keras.callbacks.LambdaCallback(on_epoch_end=testmodel) # epoch이 끝날 때 마다 testmodel함수를 호출

# repeat() : input을 무한반복 함. 한번의 epoch에 끝과 다음 epoch의 시작에 상관없이 인자 만큼 반복
history = model.fit(train_dataset.repeat(), epochs=50, steps_per_epoch=steps_per_epoch, callbacks=[testmodelcb], verbose=2)

model.save('nlp14.hdf5')

# 임의의 문장을 사용해 학습된 모델로 새로운 글 생성
test_sentence = '최참판댁 사랑은 무인지경처럼 적막하다.'

test_sentence = jamotools.split_syllables(test_sentence)
print(test_sentence)

# 앞에서 작성한 for문 복붙
next_chars = 500
for _ in range(next_chars):
    test_text_x = test_sentence[-seq_length:]
    test_text_x = np.array([char2idx[c] if c in char2idx else char2idx['UNK'] for c in test_text_x])
    test_text_x = pad_sequences([test_text_x], maxlen=seq_length, padding='pre', value=char2idx['UNK'])
    output_idx = np.argmax(model.predict(test_text_x), axis = -1)
    test_sentence += idx2char[output_idx[0]]

print('글 생성 결과 ----------')
print(jamotools.join_jamos(test_sentence))

# 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], c='r', label='loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], c='b', label='accuracy')
plt.legend()
plt.show()
