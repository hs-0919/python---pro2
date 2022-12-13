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

print(image_batch.shape) # (32, 160, 160, 3)  - 가로 세로 160 / 3은 칼라를 의미

# base model(MobileNet V2) 설계 - 대량의 데이터로 학습을 끝낸 나이스한 분류 모델

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top= False, weights='imagenet') # weights='imagenet' =>사전훈련된값을 그냥 쓸거야~ 라는 뜻.

feature_batch = base_model(image_batch)  # 해당 이미지 특징을 반환
# print(feature_batch) # shape=(32, 5, 5, 1280)

# 계층 동결
# 나이스한 모델 읽기가 끝남. - 합성곱 층 동결, 완전 연결층만 학습하는 방법을 사용
base_model.trainable = False  # 계층 동결 / base_model은 학습시키지 않음
print(base_model.summary())


# 분류 모델링(설계)
# base-model의 최종 출력 특징 : (None, 5, 5, 1280) <== 차원 축소해야함. 완전연결층에 맞도록, 즉 벡터화 작업이 필요
global_averge_layer = tf.keras.layers.GlobalAveragePooling2D()  # 공간 데이터에 대한 전역 평균 풀링 작업 -> feature를 1차원 벡터화, MaxPooling2D보다 더 급격하게 feature의 수를 줄임
feature_batch_average = global_averge_layer(feature_batch)
print(feature_batch_average.shape) # (32, 1280)  -> 5,5가 32로 축소!

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape) # (32, 1)

model = tf.keras.Sequential([
    base_model,           # 특징 추출 베이스 모델
    global_averge_layer,  # 출력값의 형태 변형을 위한 풀링 레이어
    prediction_layer      # 데이터 분석을 하는 완전 연결층
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
               loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=['accuracy'])
print(model.summary())


# 학습 전 모델 성능 확인
vali_step = 20
loss0, accuracy0 = model.evaluate(validation_batchs, steps = vali_step) # step : 기본값은 None, 평가가 1회 완료되었을음 선언하기까지의 단계(샘플배치)의 총 갯수
print('학습 전 모델 loss : {:.2f}'.format(loss0))
print('학습 전 모델 accuracy : {:.2f}'.format(accuracy0))

# 학습 
init_epochs = 10
history = model.fit(train_batchs, epochs=init_epochs, validation_data=validation_batchs)

# 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8,8))

plt.subplot(2,1,1)
plt.plot(acc, label = 'train acc')
plt.plot(val_acc, label = 'train validation acc')
plt.legend(loc='lower right')
plt.ylim([min(plt.ylim()), 1])

plt.subplot(2,1,2)
plt.plot(loss, label = 'train loss')
plt.plot(val_loss, label = 'train validation loss')
plt.legend(loc='upper right')
plt.ylim([0, 1.0])

plt.show()

# 그래프를 볼 때 모델 검증 작업의 경우 학습 횟수가 늘어나도 변동이 거의 없다.
# 학습작업의 경우 학습 횟수가 늘어나면 정확도가 조금이나마 증가하고 있다.
# 이를 볼 때 현재 모델의 예측력이 다소 만족스럽지 못하다.
# 그래서 미세조정(fine tuning)을 실시하기로 한다.

# 전이학습 방법 중 파인 튜닝 : 전이학습이 끝난 모델에 대해 마지막 레이어의 값을 일부 조정하기.
# 베이스 모델의 끝부분을 재학습 하기
base_model.trainable =True  # 학습 동결 해제
print('base model의 레이어 수 : ', len(base_model.layers))  # 54

fine_tune_at = 100  # 54개만 학습

for layer in base_model.layers[:fine_tune_at]:
    layer.trainalbe = False
    
# 모델 컴파일 후 학습
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001 / 10), # 파인튜닝 시 학습률은 1/10로 작업
               loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=['accuracy'])
print(model.summary())

# 파인튜닝 학습 
finetune_epochs = 10
total_epochs = init_epochs + finetune_epochs
history_fine = model.fit(train_batchs, epochs=total_epochs, initial_epoch=history.epoch[-1], validation_data=validation_batchs)

# 시각화
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8,8))

plt.subplot(2,1,1)
plt.plot(acc, label = 'train acc')
plt.plot(val_acc, label = 'train validation acc')
plt.legend(loc='lower right')
plt.ylim([min(plt.ylim()), 1])

plt.subplot(2,1,2)
plt.plot(loss, label = 'train loss')
plt.plot(val_loss, label = 'train validation loss')
plt.legend(loc='upper right')
plt.ylim([0, 1.0])

plt.show()

