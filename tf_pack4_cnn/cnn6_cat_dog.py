# -*- coding: utf-8 -*-
"""cnn6_cat_dog.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZkpRiZss5E1E8b1QCk28yYY7184uWzaQ
"""

# cnn을 통한 댕댕이와 냥이 분류 모델
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

# !find / -name 'cats_and_dogs*'
# !ls /root/.keras/datasets/cats_and_dogs_filtered/train/cats/ -la

# 이미지 확인
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
# print(os.listdir(train_cats_dir)[:5])  # ['cat.213.jpg', 'cat.743.jpg', 'cat.127.jpg', 'cat.979.jpg', 'cat.837.jpg']
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))
total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val
print('total train cat images : ', num_cats_tr)    # 1000
print('total train dog images : ', num_dogs_tr)    # 1000
print('total validation cat images : ', num_cats_val)  # 500
print('total validation dog images : ', num_dogs_val)  # 500
print('total train images : ', total_train)    # 2000
print('total validation images : ', total_val) # 1000

# ImageDataGenerator 클래스로 이미지 증식, 디렉토리로 레이블 작업
train_image_generator = ImageDataGenerator(rescale=1. / 255)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

# flow_from_directory() 는 인자로 설정해주는 directory의 바로 하위 디렉토리 이름을 레이블이라고 간주하고 그 레이블이라고 
# 간주한 디렉토리 아래의 파일들을 해당 레이블의 이미지들이라고 알아서 추측하여 Numpy Array Iterator를 생성
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')  # 0 or 1로 디렉토리를 라벨링
# (128, 150, 150, 3) 단위로 처리

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