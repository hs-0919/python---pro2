# Fashion MNIST로 CNN처리 - sub classing model 사용

import tensorflow as tf 
from keras import datasets, layers, models

(x_train, y_train), (x_test, y_test)= datasets.fashion_mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


x_train =x_train / 255.0
x_test = x_test / 255.0

# CNN은 채널을 사용하기 때문에 3차원 데이터를 4차원으로 변경
x_train =x_train.reshape((-1, 28, 28, 1))  # 흑백은 채널이 1개
x_test = x_test.reshape((-1, 28, 28, 1))

# model
class MyModel(models.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')
        self.conv2 = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')
        self.conv3 = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')
        self.pool = layers.MaxPool2D(pool_size=(2,2))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.2)
        self.d1 = layers.Dense(units=64, activation='relu')
        self.d2 = layers.Dense(units=32, activation='relu')
        self.d3 = layers.Dense(units=10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout(x)
        x = self.d2(x)
        x = self.dropout(x)
        return self.d3(x)

model = MyModel()

# 나머지는 이전 실습과 동일

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(x_train, y_train, batch_size=128, epochs=1000, verbose=1, validation_split=0.2,
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
# model.save('cnn3model.h5')
print()
# -------- 학습된 모델로 작업 ---------- 
# mymodel = tf.keras.models.load_model('cnn3model.h5')

# predict
import numpy as np
print('예측값 : ', np.argmax(model.predict(x_test[:1])))   # 7
print('예측값 : ', np.argmax(model.predict(x_test[[0]])))  # 7
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


