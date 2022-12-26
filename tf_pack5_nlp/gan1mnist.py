# GAN : GAN은 ‘Generative Adversarial Network’의 약자다. 
# 이 세 글자의 뜻을 풀어보는 것만으로도 GAN에 대한 전반적으로 이해할 수 있다. 
# 첫 단어인 ‘Generative’는 GAN이 생성(Generation) 모델이라는 것을 뜻한다. 
# 생성 모델이란 ‘그럴듯한 가짜’를 만들어내는 모델이다. 
# 허구 인물, 상품, 음성, 이미지 합성, 스케치로 실물영상을 생성, 텍스트 생성...

# DCGAN : (Deep Convolution GAN) - CNN을 GAN에 적용한 알고리즘 
# MNIST를 사용

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from keras.models import Sequential, Model
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import os 

if not os.path.exists("./gan_img"):
    os.makedirs("./gan_img")

np.random.seed(3)
tf.random.set_seed(3)

# 네트워크를 구성 
generator = Sequential()  # 생성자 모델 - 진짜 이미지 와 유사한 가짜 이미지 생성
generator.add(Dense(128*7*7, input_dim=100, activation=LeakyReLU(alpha=0.2))) # 이미지의 최초 크기를 준다(값이 정해진것은 아니다.) / alpha=0.2 음의 기울기 계수   
generator.add(BatchNormalization())
generator.add(Reshape((7,7,128))) # 튜플로 감싸기
generator.add(UpSampling2D()) # 이미지 2배 확대 - 이미지니까 2D
generator.add(Conv2D(64, kernel_size=5, padding='same')) # padding='same'은 원래크기 
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D()) # 다시 이미지 2배 확대 - 이미지니까 2D
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh')) # padding='same'은 원래크기 / 마지막 빠져나올때는 'tanh' 사용
print(generator.summary())

discriminator = Sequential()  # 판별자 모델
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), padding='same'))  # input_shape(28,28,1) 흑백이라 1값을 줌
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))

discriminator.add(Flatten())

discriminator.add(Dense(1, activation='sigmoid')) # 맞다 아니다 확인하기위해 'sigmoid'
print(discriminator.summary())

discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable = False  # 판별자 자신은 학습되지 않도록 기울기를 꺼줌

# 생성자와 판별자 모델을 연결시키는 GAN 모델을 생성
g_input = Input(shape=(100, ))
dis_output = discriminator(generator(g_input))
gan = Model(g_input, dis_output)

gan.compile(loss='binary_crossentropy', optimizer='adam')
print(gan.summary())

# 신경망을 실행 
def gan_train_func(epoch, batch_size, saving_interval):
    (x_train, _), (_, _) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], 28,28,1).astype('float32')
    x_train = (x_train - 127.5) / 127.5  # -1 ~ 1 사이의 실수값으로 정규화('tanh' 사용하기 때문)
    print(x_train.shape)
    true = np.ones((batch_size, 1)) # 진짜는 1
    fake = np.zeros((batch_size, 1))# 가짜는 0
    
    for i in range(epoch):
        # 실제 이미지와 판별자에 입려하는 부분
        idx = np.random.randint(0, x_train.shape[0], batch_size) # 실제 이미지를 랜덤하게 선택하기 위함
        imgs = x_train[idx]
        d_loss_real = discriminator.train_on_batch(imgs, true) # batch_size 만큼 판별시작
    
        # 가상 이미지와 판별자에 입력하는 부분 
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise) # 가짜이미지에 대한 디코딩 작업
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)  # gen_imgs에 모두 가짜(0)라는 레이블이 붙음
        
        # 판별자와 생성자의 오차를 계산
        d_loss = np.add(d_loss_real, d_loss_fake) * 0.5  # 차이를 극대화 하기위해 0.5를 곱함
        g_loss = gan.train_on_batch(noise, true) # 학습
        print('epoch:%d'%i, ', d_loss:%.4f'%d_loss, ', g_loss:%.4f'%g_loss)
        
        
        # 만들어진 이미지를 gan_img폴더에 저장 
        if i % saving_interval == 0:
            noise = np.random.normal(0, 1, (25, 100))
            gen_imgs = generator.predict(noise)
            
            # rescale images 0 ~ 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            
            # 출력
            fig, axs = plt.subplots(5,5)
            count = 0 
            for j in range(5): # 5행 5열로 출력
                for k in range(5):
                    axs[j, k].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                    axs[j, k].axis('off') # x축 y축 눈금 없앰
                    count += 1
            
            fig.savefig('./gan_img/gan_mnist_%d.png'%i) # 저장
    
gan_train_func(4001, 32, 500)  # epoch, batch_size, saving_interval
# (60000, 28, 28, 1)



