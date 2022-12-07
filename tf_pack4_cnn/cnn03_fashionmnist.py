# Fashion MNIST로 CNN처리 - Functional api 사용
# Label Description
# 0: T-shirt/top
# 1: Trouser
# 2: Pullover
# 3: Dress
# 4: Coat
# 5: Sandal
# 6: Shirt
# 7: Sneaker
# 8: Bag
# 9: Ankle boot


import tensorflow as tf 
from keras import datasets, layers, models

(x_train, y_train), (x_test, y_test)= datasets.fashion_mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


x_train =x_train / 255.0
x_test = x_test / 255.0

# CNN은 채널을 사용하기 때문에 3차원 데이터를 4차원으로 변경
x_train =x_train.reshape((-1, 28, 28, 1))  # 맨뒤가 채널수 / 모르면 -1쓰면 알아서 잡아줌 / 흑백은 채널이 1개이다.
x_test = x_test.reshape((-1, 28, 28, 1)) # 예) x_test[3, 12, 13, 1] 3번째 인덱스에서 12행, 13열에 있는 1(흑백)채널을 의미한다.

import matplotlib.pyplot as plt
# plt.figure()
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.imshow(x_train[i], cmap='gray')
# plt.show()
    

# model : functional api
input_shape = (28,28,1)
img_input = layers.Input(shape=input_shape)

net = layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(img_input)
net = layers.MaxPool2D(pool_size=(2,2))(net)

net = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(net)
net = layers.MaxPool2D(pool_size=(2,2))(net)

net = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(img_input)
net = layers.MaxPool2D(pool_size=(2,2))(net)

net = layers.Flatten()(net)

net = layers.Dense(units=64, activation='relu')(net)
net = layers.Dense(units=32, activation='relu')(net)
outputs = layers.Dense(units=10, activation='softmax')(net)

model = tf.keras.Model(inputs=img_input, outputs=outputs)

print(model.summary())

# 나머지는 이전 실습과 동일

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(x_train, y_train, batch_size=128, epochs=1000, verbose=2, validation_split=0.2,
                    callbacks=[es])

# history 저장
import pickle
history = history.history
with open('cnn3history.pickle', 'wb')as f:
    pickle.dump(history, f)


# 모델 평가
train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('train_loss:{}, train_acc:{}'.format(train_loss, train_acc))
print('test_loss:{}, test_acc:{}'.format(test_loss, test_acc))


# 모델 저장 
model.save('cnn3model.h5')
print()
# -------- 학습된 모델로 작업 ---------- 
mymodel = tf.keras.models.load_model('cnn3model.h5')

# predict
import numpy as np
print('예측값 : ', np.argmax(mymodel.predict(x_test[:1])))   # 7
print('예측값 : ', np.argmax(mymodel.predict(x_test[[0]])))  # 7
print('실제값 : ', y_test[0])

# 시각화
import matplotlib.pyplot as plt

with open('cnn3history.pickle', 'rb') as f:
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