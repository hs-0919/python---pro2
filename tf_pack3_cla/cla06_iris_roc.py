# iris dataset으로 분류 모델 여러개 생성 후 성능 비교. 최종 모델 ROC curve 표현 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense

iris = load_iris()
print(iris.keys())

x = iris.data
print(x[:2])
y = iris.target
print(y)

names = iris.target_names  # 꽃의 종류명
print(names)  # ['setosa' 'versicolor' 'virginica']
feature_names = iris.feature_names
print(feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# label에 대해 원핫 인코딩
print(y[:1], y.shape) # [0] (150,) , [[1. 0. 0.]] (150, 3)
onehot = OneHotEncoder(categories='auto') # sklearn: OneHotEncoder(), keras: to_categorical, numpy: eye(), pandas: get_dummies()
y = onehot.fit_transform(y[:, np.newaxis]).toarray() # 차원확대
print(y[:1], y.shape)

# feature에  대해 표준화
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
print(x_scale[:2])

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.3, random_state=1)

n_features = x_train.shape[1]
n_classes = y_train.shape[1]
print('feature 수 : {}, label수 : {} '.format(n_features, n_classes)) # 4, 3

print('----model ----')
def create_model_func(input_dim, output_dim, out_nodes, n, model_name='model'):
    # print(input_dim, output_dim, out_nodes, n, model_name)
    def create_model():
        model = Sequential(name=model_name)
        for _ in range(n):
            model.add(Dense(units= out_nodes, input_dim=input_dim, activation='relu'))
        
        model.add(Dense(units=output_dim, activation='softmax')) # 출력층
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        return model
    return create_model
        
models = [create_model_func(n_features, n_classes, 10, n, 'model_{}'.format(n)) for n in range(1,4)] 
print(len(models))

for cre_model in models:
    print()
    cre_model().summary()

history_dict = {}
for cre_model in models:
    model = cre_model()
    print('모델명 : ', model.name)
    histories = model.fit(x_train, y_train, batch_size=5, epochs=50, validation_split=0.3, verbose=0)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('test loss : ', score[0])
    print('test acc : ', score[1])
    history_dict[model.name] = [histories, model]
    
print(history_dict)

# 모델 성능을 확인 시각화
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

for model_name in history_dict:
    print('history_dict : ', history_dict[model_name][0].history['acc'])
    
    val_acc = history_dict[model_name][0].history['val_acc']
    val_loss = history_dict[model_name][0].history['val_loss']
    ax1.plot(val_acc, label=model_name)
    ax2.plot(val_loss, label=model_name)
    ax1.set_ylabel('val_acc')
    ax2.set_ylabel('val_loss')
    ax2.set_ylabel('val_loss')
    ax2.set_xlabel('epochs')
    ax1.legend()
    ax2.legend()
plt.show()

# ROC curve로 모델 성능 확인
from sklearn.metrics import roc_curve, auc

plt.figure()
plt.plot([0,1], [0,1], 'k--')

for model_name in history_dict:
    model= history_dict[model_name][1]
    y_pred = model.predict(x_test)
    # fpr, tpr 구하기 
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel()) # 차원 축소
    plt.plot(fpr, tpr, label='{}, AUC value : :{:.3f}'.format(model_name, auc(fpr, tpr)))
    
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()

print()
# k-fold 교차검증 수행하여 모델 성능 비교
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

create_model = create_model_func(n_features, n_classes, 10, 1)
estimator = KerasClassifier(build_fn=create_model, epochs=50, batch_size=10, verbose=0)
scores = cross_val_score(estimator, x_scale, y, cv=10)
print('acc1 : {:0.2f} (+/-{:0.2f})'.format(scores.mean(), scores.std()))

create_model = create_model_func(n_features, n_classes, 10, 2)
estimator = KerasClassifier(build_fn=create_model, epochs=50, batch_size=10, verbose=0)
scores = cross_val_score(estimator, x_scale, y, cv=10)
print('acc2 : {:0.2f} (+/-{:0.2f})'.format(scores.mean(), scores.std()))

create_model = create_model_func(n_features, n_classes, 10, 3)
estimator = KerasClassifier(build_fn=create_model, epochs=50, batch_size=10, verbose=0)
scores = cross_val_score(estimator, x_scale, y, cv=10)
print('acc3 : {:0.2f} (+/-{:0.2f})'.format(scores.mean(), scores.std()))

print('------------------')
# 위 작업 후 가장 좋은 모델을 확인 후 최종 모델을 작성 .. 



