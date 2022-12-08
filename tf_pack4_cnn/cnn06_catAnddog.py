# CNN을 통한 댕댕이 와 냥냥이 분류 모델

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

data_url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=data_url, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

# 상수 정의
batch_size = 128
epochs=15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# 데이터 준비
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# 이미지 확인
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
# print(os.listdir(train_cats_dir)[:5])
# ['cat.0.jpg', 'cat.1.jpg', 'cat.10.jpg', 'cat.100.jpg', 'cat.101.jpg']
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))
total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
print('total train cat images : ', num_cats_tr)  # 1000
print('total train dog images : ', num_dogs_tr)  # 1000
print('total validation cat images : ', num_cats_val)  # 500
print('total validation dog images : ', num_dogs_val)  # 500
print('total train images : ', total_train)    # 2000
print('total validation images : ', total_val) # 1000


# ImageDataGenerator 클래스로 이미지 증식, 디렉토리로 레이블 작업
train_image_generator = ImageDataGenerator(rescale=1. / 255)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

# flow_from_directory() 는 인자로 설정해주는 directory의 바로 하위 디렉토리 이름을 레이블이라고 간주하고 
# 그 레이블이라고 간주한 디렉토리 아래의 파일들을 해당 레이블의 이미지들이라고 알아서 추측하여 Numpy Array Iterator를 생성한다. 
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')  # 0 or 1로 디렉토리를 라벨링

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=validation_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

# 데이터 확인
sample_train_images, _ = next(train_data_gen)

def plotImage_func(images_arr):    # 1행 5열
    fig, axes = plt.subplots(1, 5, figsize=(10, 20))
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.show()

plotImage_func(sample_train_images[:5])

# model
model = Sequential([
    Conv2D(filters=16, kernel_size=3, strides=1, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'),
    MaxPooling2D(pool_size=2), # pool_size=(2,2) default값
    Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
print(model.summary())

# flow_from_directory() 사용해 레이블 입력을 대신해야 하므로...
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,  # 하나의 에폭을 처리하고 다음 에폭을 시작하기 전까지 generator에서 생성할 단계의(샘플배치) 총갯수
    epochs=epochs,
    validation_data = val_data_gen,
    validation_steps = total_val // batch_size

) 

model.save('catdog.h5')

# 학습 결과 시각화 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch_range = range(epochs)

plt.figure(figsize=(10, 8))

plt.subplot(1,2,1)
plt.plot(epoch_range, acc, label='train acc')
plt.plot(epoch_range, val_acc, label='train val_acc')
plt.legend(loc='best')

plt.subplot(1,2,2)
plt.plot(epoch_range, loss, label='train loss')
plt.plot(epoch_range, val_loss, label='train val_loss')
plt.legend(loc='best')
plt.show()

# 과적합 발생
# 원인 : 데이터 수 부족 의심 - 데이터 보강을 해보자
image_gen_train = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range = 30,
    width_shift_range= 15,
    height_shift_range= 15,
    horizontal_flip=True,
    zoom_range=0.5
)

train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')  # 0 or 1로 디렉토리를 라벨링

# 보강 이미지 시각화
augmented_img = [train_data_gen[0][0][0] for i in range(5)] 
plotImage_func(augmented_img)


image_gen_val = ImageDataGenerator(
    rescale = 1. / 255
)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')  # 0 or 1로 디렉토리를 라벨링

# new_model
new_model = Sequential([
    Conv2D(filters=16, kernel_size=3, strides=1, padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), activation='relu'),
    MaxPooling2D(pool_size=2), # pool_size=(2,2) default값
    Dropout(rate=0.2),
    Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(rate=0.2),
    Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(rate=0.2),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=1)
])

new_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
print(new_model.summary())

# 새로운 모델로 학습 

history = new_model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,  # 하나의 에폭을 처리하고 다음 에폭을 시작하기 전까지 generator에서 생성할 단계의(샘플배치) 총갯수
    epochs=epochs,
    validation_data = val_data_gen,
    validation_steps = total_val // batch_size

) 

new_model.save('catdog.h5')

# 학습 결과 시각화 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch_range = range(epochs)

plt.figure(figsize=(10, 8))

plt.subplot(1,2,1)
plt.plot(epoch_range, acc, label='train acc')
plt.plot(epoch_range, val_acc, label='train val_acc')
plt.legend(loc='best')

plt.subplot(1,2,2)
plt.plot(epoch_range, loss, label='train loss')
plt.plot(epoch_range, val_loss, label='train val_loss')
plt.legend(loc='best')
plt.show()

# 새로운 이미지 분류 - 콜랩에서 실행

# from google.colab import files
from keras.preprocessing import image

mymodel =tf.keras.models.load_model('catdog.h5')
# uploaded = files.upload()

# for fn in uploaded.keys():
#   path='/content/'+fn 
#   img=tf.keras.utils.load_img(path, target_size=(150,150))
#
#   x = tf.keras.utils.img_to_array(img)
#   x = np.expand_dims(x, axis=0)
#   # print(x)
#
#   images =np.vstack([x])
#   # print(images)

# classes = mymodel.predict(images, batch_size=10)
# print(classes)
#
# if classes[0] > 0:
#   print(fn + '너는 댕댕이')
# else:
#   print(fn + '너는 냥냥이')








