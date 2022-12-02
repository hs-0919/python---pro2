# red & white wine dataset으로 분류 모델 작성

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

wdf = pd.read_csv("../testdata/wine.csv", header=None)
print(wdf.head(2))
print(wdf.info())
print(wdf.iloc[:, 12].unique())  # [1 0]
print(len(wdf[wdf.iloc[:, 12] == 0])) # 4898
print(len(wdf[wdf.iloc[:, 12] == 1])) # 1599

dataset = wdf.values
x =dataset[:, 0:12]
y =dataset[:, -1]
print(x[:1])
print(y[:1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)

# model 
model =Sequential()
model.add(Dense(units=32, input_dim=12, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 훈려하지 않고 평가
loss, acc = model.evaluate(x=x_train, y=y_train, verbose=2)
print('loss, acc : ', loss, acc)

# 학습 조기 종료
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# 모델 학습 시 모니터링 결과를 파일로 저장 
chkpoint = ModelCheckpoint(filepath='cl3model.hdf5', monitor='val_loss', verbose=0, save_best_only=True)

history = model.fit(x=x_train, y=y_train, epochs=1000, batch_size=32, verbose=2,
                     validation_split=0.2,
                     callbacks=[early_stop, chkpoint])

# 훈려하고 평가
loss, acc = model.evaluate(x=x_test, y=y_test, verbose=2)
print('loss, acc : ', loss, acc)  # loss, acc :  0.2001439779996872 0.9312820434570312

vloss = history.history['val_loss']
loss = history.history['loss']
print('vloss : ', vloss)
print('loss : ', loss)

vaccuracy = history.history['val_accuracy']
accuracy = history.history['accuracy']
print('vaccuracy : ', vaccuracy)
print('accuracy : ', accuracy)

# 시각화
epoch_len =np.arange(len(accuracy))

plt.plot(epoch_len, vloss, c='red', label='val_loss')
plt.plot(epoch_len, loss, c='blue', label='loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

plt.plot(epoch_len, vaccuracy, c='red', label='val_accuracy')
plt.plot(epoch_len, accuracy, c='blue', label='accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()


print()
# best model을 읽어 새로운 자료 분류 
from keras.models import load_model
mymodel = load_model('cl3model.hdf5')
new_data = x_test[:5, :]
print(new_data)
pred = mymodel.predict(new_data)
print('pred : ', np.where(pred > 0.5, 1, 0).flatten())

