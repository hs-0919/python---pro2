# 케라스에서 제공하는 로이터 뉴스 데이터를 LSTM을 이용하여 텍스트 분류를 진행해보겠습니다. 
# 로이터 뉴스 기사 데이터는 총 11,258개의 뉴스 기사가 46개의 뉴스 카테고리로 분류되는 뉴스 기사 데이터입니다.

import numpy as np
from keras.datasets import reuters
from keras.utils import pad_sequences, np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt

max_features = 10000

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(len(set(y_train)))
print(x_train[10])
print(y_train[10])

# train / validation split
x_val = x_train[7000:]
y_val = y_train[7000:]
x_train = x_train[:7000]
y_train = y_train[:7000]

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

# 문장 길이 맞추기 
text_max_words = 120
x_train = pad_sequences(x_train, maxlen=text_max_words)
x_val = pad_sequences(x_val, maxlen=text_max_words)
x_test = pad_sequences(x_test, maxlen=text_max_words)
print(x_train[0], len(x_train[0]))

# onehot처리
y_train=np_utils.to_categorical(y_train)
y_val=np_utils.to_categorical(y_val)
y_test=np_utils.to_categorical(y_test)
print(y_train[0])


# 모델 구성 방법 1 : Dense로만 구성
model = Sequential()
model.add(Embedding(max_features, 128, input_length=text_max_words))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(46, activation='softmax'))
print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


hist = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val), verbose=2)

# 시각화 
def plt_func():
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    loss_ax.plot(hist.history['loss'], c='y', label = 'train loss')
    loss_ax.plot(hist.history['val_loss'], c='r', label = 'val loss')
    loss_ax.set_ylim([0, 3])
    
    acc_ax.plot(hist.history['accuracy'], c='y', label = 'train accuracy')
    acc_ax.plot(hist.history['val_accuracy'], c='r', label = 'val accuracy')
    acc_ax.set_ylim([0, 1])
    
    
    loss_ax.legend()
    acc_ax.legend()
    plt.show()
    
    loss_metrics = model.evaluate(x_test, y_test, batch_size=64)
    print(loss_metrics)
    
    
plt_func()
