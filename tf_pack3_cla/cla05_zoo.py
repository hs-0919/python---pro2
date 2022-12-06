# zoo animal dataset으로 동물의 type을 분류

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical


xy = np.loadtxt("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/zoo.csv", delimiter=',')
print(xy[0])
print(xy.shape) # (101, 17)

x_data = xy[:, 0:-1]
y_data = xy[:, -1]
print(x_data[0])
print(y_data[0], '', set(y_data)) # {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

# train / test는 생략

# label은 one-hot처리를 해야 함. 
# y_data = to_categorical(y_data)
# print(y_data[0])

model = Sequential()
model.add(Dense(32, input_shape=(16,), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))
print(model.summary())

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# loss='sparse_categorical_crossentropy' 하면 내부적으로 one-hot처리를 한다.

history = model.fit(x_data, y_data, epochs=20, batch_size=10, validation_split=0.3, verbose=0)
print('evaluate: ', model.evaluate(x_data, y_data, batch_size=10, verbose=0))

# loss, acc 시각화
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

import matplotlib.pyplot as plt
plt.plot(loss, 'b-', label='loss')
plt.plot(val_loss, 'r--', label='val_loss')
plt.xlabel('epochs')
plt.legend()
plt.show()

plt.plot(acc, 'b-', label='acc')
plt.plot(val_loss, 'r--', label='val_acc')
plt.xlabel('epochs')
plt.legend()
plt.show()


print()
# 한 개 예측 
pred_data = x_data[:1]
print(model.predict(pred_data))
print('예측값 : ', np.argmax(model.predict(pred_data)))


print()
# 여러 개 값 예측하기
pred_datas = x_data[:5]
preds = [np.argmax(i) for i in model.predict(pred_datas)]
print('예측값 : ', preds)
print('실제값 : ', y_data[:5])

