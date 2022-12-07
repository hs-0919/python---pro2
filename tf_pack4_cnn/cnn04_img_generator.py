# 이미지 보강 - 이미지가 부족한 경우 기존 이미지를 변형시켜 이미지 수를 늘림

import tensorflow as tf 
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test)= fashion_mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# print(y_train[:3])
y_train = to_categorical(y_train)
# print(y_train[:3])
y_test = to_categorical(y_test)

# plt.figure(figsize=(10,10))
# for c in range(100):
#     plt.subplot(10, 10, c + 1)
#     plt.axis('off') # x, y축 안보이게
#     plt.imshow(x_train[c].reshape(28, 28), cmap='gray')
# plt.show()

# 이미지 보강 연습 : x_train[0] - Ankle boot
from keras.preprocessing.image import ImageDataGenerator

img_generator = ImageDataGenerator(
    # rotation_range : 이미지 회전값
    # zoom_range : 이미지 일부확대
    # shear_range : 이미지 기울기
    # width_shift_range : 좌우(수평) 이동
    # height_shift_range : 상하(수직) 이동
    # horizontal_flip : 이미지 가로뒤집기
    # vertical_flip : 이미지 세로뒤집기
    rotation_range=10,
    zoom_range=0.1,
    shear_range=0.5,
    width_shift_range= 0.1, 
    height_shift_range= 0.1,
    horizontal_flip= True,
    vertical_flip= True
)

augument_size = 100
x_augument = img_generator.flow(np.tile(x_train[0].reshape(28*28), 100).reshape(-1, 28, 28, 1),
                                np.zeros(augument_size),
                                batch_size = augument_size,
                                shuffle = False).next()[0]
# print(x_augument.shape) # (100, 28, 28, 1)
"""
# 원본 이미지에 3만개의 이미지를 추가 ---------------
img_generator = ImageDataGenerator(
    rotation_range=20,   # 이미지 회전 값
    zoom_range=0.5,   # 이미지일부 확대/축소
    shear_range=0.5,   # 이미지 기울기로 회전
    width_shift_range = 0.2,   # 좌우(수평) 이동
    height_shift_range = 0.1,  # 상하(수직) 이동
    horizontal_flip = True,    # 이미지 가로 뒤집기
    vertical_flip = True       # 이미지 세로 뒤집기
)

augument_size = 30000
np.random.seed(0)
tf.random.set_seed(0)
randinx = np.random.randint(x_train.shape[0], size=augument_size)  # 인덱스로 사용할 난 수 얻기
print(randinx)
x_augment = x_train[randinx].copy()   # 랜덤하게 원본에서 30000 개 추출
y_augment = y_train[randinx].copy() 

x_augment = img_generator.flow(x_augment,
                               np.zeros(augument_size),
                               batch_size=augument_size,
                               shuffle=False).next()[0]
print(x_augment.shape)  # (30000, 28, 28, 1)
# 원래 데이터인 x_train에 이미지 증강 결과를 합침
x_train = np.concatenate((x_train, x_augment))
y_train = np.concatenate((y_train, y_augment))
print(x_train.shape)  # (90000, 28, 28, 1)
print(y_train.shape)  # (90000, 10)
# ------------------------------------------------
"""
'''
plt.figure(figsize=(10,10))
for c in range(100):
    plt.subplot(10,10,c+1)
    plt.axis('off') # x축y축 다 빼버리기
    plt.imshow(x_augument[c].reshape(28,28), cmap='gray')
plt.show()
'''

# model
model = tf.keras.models.Sequential([
    # 이미지 데이터 특징 추출로 크기, 용량을 줄임
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=(28,28,1),
                           padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    
    # 벡터로 데이터 할 줄로 세우기
    tf.keras.layers.Flatten(),  
    
    # 이미지 분류기
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax'),
    
])

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

import os
MODEL_DIR = './mymnist/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = './mymnist/{epoch:02d}-{val_loss:.3f}.hdf5'
chkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=64, verbose=2,
                    callbacks=[es, chkpoint])
print('evaluate acc : %.4f'%(model.evaluate(x_test, y_test, batch_size=64, verbose=0)[1]))

history = history.history
plt.subplot(1,2,1)
plt.plot(history['acc'], marker='o', c='green' , label='accuracy')
plt.plot(history['val_acc'], marker='s', c='blue', label='val_accuracy')
plt.xlabel('epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history['loss'], marker='o', c='green' , label='loss')
plt.plot(history['val_loss'], marker='s', c='blue', label='val_loss')
plt.xlabel('epochs')
plt.legend()

plt.show()