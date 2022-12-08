# 전이 학습 : 학습 데이터가 부족한 분야의 모델 구축을 위해 데이터가 풍부한 분야에서 
#           훈련된 모델을 재사용하는 머신러닝 학습 기법

# 구글이 만든 MobileVet V2모델을 사용
# pip install tensorflow-dataset - 아나콘다 프롬푸트! / 콜랩에도 설치후 사용


# MobileVet V2 모델을 일부 재학습한 후  개/고양이 분류 모델을 생성
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


(raw_train, raw_validation, raw_test),metadata  = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info=True, as_supervised=True)

print(raw_train)
print(raw_validation)
print(raw_test)


from tensorflow_datasets.core.dataset_info import Metadata
get_label_name = metadata.features['label'].int2str

# 개 고양이 이미지 불러옴
for image, label in raw_train.take(10):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()

# image formating : MobileVet V2 모델이 원하기 때문에
IMG_SIZE = 160

def format_exam(image, label):  # dataset 함수로 넣어주기 위함
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_exam)
validation = raw_validation.map(format_exam)
test = raw_test.map(format_exam)

# 이미지 섞기
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE =1000

train_batchs = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)  # train만 shuffle 하기
validation_batchs = validation.batch(BATCH_SIZE)
test_batchs = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batchs.take(1):
    pass

print(image_batch.shape) # (32, 160, 160, 3)

# base model(MobileNet V2) 설계 - 대량의 데이터로 학습을 끝낸 나이스한 분류 모델

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top= False, weights='imagenet') # weights='imagenet' =>사전훈련된값을 그냥 쓸거야~ 라는 뜻.

feature_batch = base_model(image_batch)  # 해당 이미지 특징을 반환
# print(feature_batch) # shape=(32, 5, 5, 1280)

# 계층 동결
# 나이스한 모델 읽기가 끝남. - 합성곱 층 동결, 완전 연결층만 학습하는 방법을 사용
base_model.trainable = False  # 계층 동결 / base_model은 학습시키지 않음
print(base_model.summary())


