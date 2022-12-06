# 내가 그린 손글씨 이미지 분류 결과 확인

import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image    # 이미지 확대 / 축소 기능
import tensorflow as tf

im = Image.open('number.png')
img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert('L'))  
# ANTIALIAS - 저해상도를 고해상도로 이미지 부드럽게 ,   convert('L') - 그레이스케일, 흑백?
print(img.shape) # (28, 28)

# plt.imshow(img, cmap='Greys')
# plt.show()


# (28, 28) -> 784열 짜리로
data = img.reshape([1, 784])
# print(data)
data = data/255.0  # 정규화 (∵ 모델 학습 시 정규화를 선행 했으므로)
# print(data) #   0.2745098  0.         0.01176471 0.         0.         0. ... 

# 학습이 끝난 모델로 내 이미지를 판별  (cnn이 이미지 판별은 아니다...)
mymodel = tf.keras.models.load_model('cla07model.hdf5')
pred = mymodel.predict(data)
print('pred : ', np.argmax(pred, 1))




