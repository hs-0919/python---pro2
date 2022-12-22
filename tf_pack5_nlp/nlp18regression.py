# LSTM을 사용한 삼성전자 주가 예측 (종가)
# KRX: 005930
# !pip install finance-datareader

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import FinanceDataReader as fdr


STOCK_CODE = '005930'
stock_data = fdr.DataReader(STOCK_CODE)
print(stock_data.head())
print(stock_data.tail())

print('상관관계 : \n', stock_data.corr(method='pearson'))

stock_data.reset_index(inplace=True)
stock_data.drop(['Change'], axis='columns', inplace=True)
print(stock_data.head(3))
print(stock_data.tail(3))
print(stock_data.info())

# Date열을 연, 월, 일 로 분리 
stock_data['year'] = stock_data['Date'].dt.year
stock_data['month'] = stock_data['Date'].dt.month
stock_data['day'] = stock_data['Date'].dt.day
print(stock_data.head(3))
print(stock_data.shape)  # (6000, 9)

# 1998년 이후 주가 흐름 시각화 
df = stock_data.loc[stock_data['year'] >= 1998]
plt.figure(figsize=(6,4))
sns.lineplot(y=df['Close'], x=df.year)
plt.xlabel('year')
plt.ylabel('Close')
plt.show()

# 스케일링 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols =['Open', 'High', 'Low', 'Close', 'Volume']
df_scaled = scaler.fit_transform(stock_data[scale_cols])

df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols
print(df_scaled.head(3))

only_close = ['Close']
close_scaled = scaler.fit_transform(stock_data[only_close])  # predict을 위함
print('스케일 값 : ', close_scaled[:5].ravel())
print('복원 값 : ', scaler.inverse_transform(close_scaled[:5]).ravel())
print('최초 값 : ', stock_data['Close'].values[:5])

# 이전 20일을 기준으로 다음날 종가 예측
TEST_SIZE = 200  # 학습은 200일 기준으로 
train = df_scaled[:-TEST_SIZE]  # 관찰값 처음부터 200일 이전 데이터
test = df_scaled[-TEST_SIZE:]  # 200일 이후 데이터
print(train.shape)  # (5800, 5)
print(test.shape)   # (200, 5)

def make_dataset_func(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i + window_size]))
        label_list.append(np.array(label.iloc[i + window_size]))
    return np.array(feature_list), np.array(label_list)


# feature, label 정하기
feature_cols = ['Open', 'High', 'Low', 'Volume']
label_cols = ['Close']

train_feature = train[feature_cols]
train_label = train[label_cols]
test_feature = test[feature_cols]
test_label = test[label_cols]

train_feature, train_label = make_dataset_func(train_feature, train_label, 20)
print(train_feature[:2])
print(train_label[:2])
print(train_feature.shape, train_label.shape) # (5780, 20, 4) (5780, 1)

# train / test split 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_feature, train_label, test_size=0.2, shuffle=False, random_state=12 )
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (4624, 20, 4) (1156, 20, 4) (4624, 1) (1156, 1)
test_feature, test_label = make_dataset_func(test_feature, test_label, 20)


from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(LSTM(units=16, activation='tanh', input_shape=(train_feature.shape[1], train_feature.shape[2]), return_sequences=False ))
model.add(Dense(16, activation='relu'))
model.add(Dense(1,activation='linear'))

# 'mse'는 이상치에 민감하다. - Huber loss 는 모든 지점에서 미분이 가능하면서 이상치에 강건한(robust) 성격을 보인다.
from keras.losses import Huber
loss =Huber()
model.compile(optimizer='adam', loss=loss, metrics=['mse'])

es = EarlyStopping(monitor='val_loss', mode='auto', patience=3)
model_chk = ModelCheckpoint('nlp18.h5', monitor='val_loss', save_best_only=True, verbose=0)
history = model.fit(x_train, y_train, epochs=50, batch_size=8, validation_data=(x_test, y_test), verbose=2, callbacks=[es, model_chk])

# 시각화
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss') # 둘의 거리가 멀 수록 문제가 있다.
plt.legend()
plt.show()

# predict 
from sklearn.metrics import r2_score
pred = model.predict(test_feature, verbose=0)
print('r2_score : ', r2_score(test_label, pred))

print('pred : ', np.round(pred[:10].flatten(), 2))
print('pred(스케일 원복) : ', scaler.inverse_transform(pred[:10]).flatten())
print('real(스케일 원복) : ', scaler.inverse_transform(test_label[:10]).flatten())

# 시각화
plt.figure(figsize=(6,4))
plt.plot(test_label[:20], label='real')
plt.plot(pred[:20].flatten(), label='pred') # 둘의 거리가 멀 수록 문제가 있다.
plt.legend()
plt.show()


