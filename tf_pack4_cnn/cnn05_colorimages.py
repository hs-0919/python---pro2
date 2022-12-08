# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
# There are 50000 training images and 10000 test images.
# 
# airplane automobile bird cat deer dog frog horse ship truck 

import numpy as np
import matplotlib.pyplot as  plt
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape) # (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)

print(x_train[0])
print(y_train[0])
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(x_train[0])
plt.subplot(132)
plt.imshow(x_train[1])
plt.subplot(133)
plt.imshow(x_train[2])
plt.show()

x_train = x_train.astype('float32') / 255  # 정규화 시키기
x_test = x_test.astype('float32') / 255

NUM_CLASSES = 10
y_train = to_categorical(y_train, NUM_CLASSES)  # 레이블에 대해 원핫 처리
y_test = to_categorical(y_test, NUM_CLASSES)
# print(y_train[0])

# model : CNN X
'''
model = Sequential([
    Dense(units=512, activation='relu', input_shape=(32, 32, 3)),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=NUM_CLASSES, activation='softmax')
])
print(model.summary())
'''

# functional api 
from keras import optimizers
input_layer = Input((32,32,3))
x = Flatten()(input_layer)
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(input_layer, output_layer)
print(model.summary())

opti = Adam(learning_rate=0.01)
model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10, shuffle=True, verbose=2)
print('test acc : %.4f'%(model.evaluate(x_test, y_test, verbose=0, batch_size=128)[1]))
print('train acc : %.4f'%(model.evaluate(x_train, y_train, verbose=0, batch_size=128)[1]))
print('test loss : %.4f'%(model.evaluate(x_test, y_test, verbose=0, batch_size=128)[0]))

# CNN을 네트워크에 추가

from keras import optimizers
from keras.layers import Conv2D, Activation, ReLU, LeakyReLU, BatchNormalization, MaxPool2D
# functional api 
input_layer = Input((32,32,3))
x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(input_layer)
# x = ReLU()(x)
x = LeakyReLU()(x)
x = MaxPool2D(pool_size=2)(x)

x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
x = LeakyReLU()(x)
x = MaxPool2D(pool_size=2)(x)

x = Flatten()(input_layer)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)  # gradient 폭주를 방지, 과적합 방지도 가능(Dropout과 함께 적어주는 경우는 별로 없다.)
x = Dense(128, activation='relu')(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(input_layer, output_layer)
print(model.summary())

opti = Adam(learning_rate=0.01)
model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10, shuffle=True, verbose=2)

print('test acc : %.4f'%(model.evaluate(x_test, y_test, verbose=0, batch_size=128)[1]))
print('train acc : %.4f'%(model.evaluate(x_train, y_train, verbose=0, batch_size=128)[1]))
print('test loss : %.4f'%(model.evaluate(x_test, y_test, verbose=0, batch_size=128)[0]))

pred = model.predict(x_test[:10])
import numpy as np
print('예측값 : ', np.argmax(pred, axis=-1))
print('실제값 : ', np.argmax(y_test[:10], axis=-1))