# MNIST로 CNN처리 
# 1) Conv(이미지 특징 추출) + Pooling(Conv 결과를 샘플링 - Conv 결과인 Feature map의 크기를 다시 줄임)
# 2) 원래의 이미지 크기를 줄여 최종적으로 FCLayer를 진행 (Conv + Pooling 결과를 다차원 배열 자료를 1차원으로 만들어 한 줄로 세우기)
# 3) Dense 층으로 넘겨 분류 작업 수행
import tensorflow as tf 
from keras import datasets, layers, models

(x_train, y_train), (x_test, y_test)= datasets.mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

# CNN 사용 -> 콜랩 가기
# CNN은 채널을 사용하기 때문에 3차원 데이터를 4차원으로 변경
x_train =x_train.reshape((60000, 28, 28, 1))  # 맨뒤가 채널수 / 모르면 -1쓰면 알아서 잡아줌 / 흑백은 채널이 1개이다.
x_test = x_test.reshape((10000, 28, 28, 1)) # 예) x_test[3, 12, 13, 1] 3번째 인덱스에서 12행, 13열에 있는 1(흑백)채널을 의미한다.
# print(x_train.ndim)
# print(x_train[:1])

x_train =x_train / 255.0
x_test = x_test / 255.0
print(x_train[:1])

# label의 원핫 처리는 모델에게 위임 / 원핫 처리 해야하지만 귀찮잔하~~
# model
input_shape = (28, 28, 1)

# Sequential api 사용
model = models.Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='valid',
                        activation='relu', input_shape=input_shape)) # kernel_size=(3,3), strides=(1,1) => 기본값
# strides=(1,1) -> 한칸씩 이동 (2,2) -> 두칸씩 이동 / strides=(1,1)은 생략 가능함
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(rate=0.2)) # 과적합 방지.

model.add(layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')) 
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(rate=0.2)) # 과적합 방지.

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')) 
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(rate=0.2)) # 과적합 방지.

model.add(layers.Flatten())  # FCLayer(Fully Connected Layer) : 2차원을 1차원으로 변경(모든 배열 자료를 한 줄로 세움)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(rate=0.2)) # 과적합 방지.
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(rate=0.2)) # 과적합 방지.
model.add(layers.Dense(10, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(x_train, y_train, batch_size=128, epochs=1000, verbose=2, validation_split=0.2,
                    callbacks=[es])

# history 저장
import pickle
history = history.history
with open('cnn2history.pickle', 'wb')as f:
    pickle.dump(history, f)


# 모델 평가
train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('train_loss:{}, train_acc:{}'.format(train_loss, train_acc))
print('test_loss:{}, test_acc:{}'.format(test_loss, test_acc))


# 모델 저장 
model.save('cnn2model.h5')
print()
# -------- 학습된 모델로 작업 ---------- 
mymodel = tf.keras.models.load_model('cnn2model.h5')

# predict
import numpy as np
print('예측값 : ', np.argmax(mymodel.predict(x_test[:1])))   # 7
print('예측값 : ', np.argmax(mymodel.predict(x_test[[0]])))  # 7
print('실제값 : ', y_test[0])

# 시각화
import matplotlib.pyplot as plt

with open('cnn2history.pickle', 'rb') as f:
    history = pickle.load(f)

def plot_acc_func(title=None):
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel(title)
    plt.legend()

plot_acc_func('accuracy')
plt.show()

def plot_loss_func(title=None):
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel(title)
    plt.legend()

plot_loss_func('loss')
plt.show()
