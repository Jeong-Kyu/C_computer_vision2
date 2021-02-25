import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 늘리는 작업
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras import Sequential,Model
from keras.layers import *
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, MaxPooling2D, Conv2D, Flatten, MaxPooling1D, Conv1D,Dropout, BatchNormalization, AveragePooling2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import PIL.Image as pilimg

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('./dirty_mnist_data/mnist_data/train.csv')
test = pd.read_csv('./dirty_mnist_data/mnist_data/test.csv')

# drop 인덱스
train2 = train.drop(['id','digit','letter'],1) # 인덱스 있는 3개 버리기
test2 = test.drop(['id','letter'],1) #인덱스 있는 것 버리기

# convert pandas dataframe to numpy array
train2 = train2.values
test2 = test2.values
# print(train2.shape) #(2048, 784)
# print(test2.shape) # (20480, 784)

# 정규화(Minmax도 해보기) ---> standard보다 Minmax가 잘나온다
scaler = MinMaxScaler()
scaler.fit(train2)
scaler.transform(train2)
scaler.transform(test2)

# # reshape
train2 = train2.reshape(-1,28,28,1)
test2 = test2.reshape(-1,28,28,1)
# train2 = train2.reshape(-1,97,2,1)
# test2 = test2.reshape(-1,97,2,1) #4차원

# ImageDatagenerator & data augmentation
idg = ImageDataGenerator(height_shift_range=(-1,1),width_shift_range=(-1,1)) # 이미지 카테고리화(4차원만 가능)
idg2 = ImageDataGenerator() #ImageDataGenerator 머신러닝
# width_shift_range 좌우로 움직이는 정도:(-1,1) 처음부터 끝까지
# height_shift_range 위아래로 움직이는 정도
# ===============================================================================================================================

# File Load
train0 = pd.read_csv('./dirty_mnist_data/dirty_mnist_2nd_answer.csv')
print(train.shape)  # (50000, 27)

sub0 = pd.read_csv('./dirty_mnist_data/sample_submission.csv')
print(sub0.shape)    # (5000, 27)

######################################################

#1. DATA

#### train
df_x = []

for i in range(0,50000):
    if i < 10 :
        file_path = './dirty_mnist_data/dirty_mnist_2nd/0000' + str(i) + '.png'
    elif i >=10 and i < 100 :
        file_path = './dirty_mnist_data/dirty_mnist_2nd/000' + str(i) + '.png'
    elif i >= 100 and i <1000 :
        file_path = './dirty_mnist_data/dirty_mnist_2nd/00' + str(i) + '.png'
    elif i >= 1000 and i < 10000 :
        file_path = './dirty_mnist_data/dirty_mnist_2nd/0' + str(i) + '.png'
    else : 
        file_path = './dirty_mnist_data/dirty_mnist_2nd/' + str(i) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_x.append(pix)

x = pd.concat(df_x)
x = x.values
print("x.shape ", x.shape)       # (12800000, 256) >>> (50000, 256, 256, 1)
print(x[0,:])
x = x.reshape(50000, 256, 256, 1)
x = x/255
x = x.astype('float32')

y = train.iloc[:,1:]
print("y.shape ", y.shape)    # (50000, 26)

np.save('../computervision2/dirty_x.npy', arr=x)
np.save('../computervision2/dirty_y.npy', arr=y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=47)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=47)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ImageGenerator >> 그림 확인하기
idg = ImageDataGenerator(
    height_shift_range=0.1,
    width_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.2
    )
idg2 = ImageDataGenerator()

#### pred
df_pred = []

for i in range(0,5000):
    if i < 10 :
        file_path = './dirty_mnist_data/test_dirty_mnist_2nd/5000' + str(i) + '.png'
    elif i >=10 and i < 100 :
        file_path = './dirty_mnist_data/test_dirty_mnist_2nd/500' + str(i) + '.png'
    elif i >= 100 and i <1000 :
        file_path = './dirty_mnist_data/test_dirty_mnist_2nd/50' + str(i) + '.png'
    else : 
        file_path = './dirty_mnist_data/test_dirty_mnist_2nd/5' + str(i) + '.png'
    image = pilimg.open(file_path)
    pix = np.array(image)
    pix = pd.DataFrame(pix)
    df_pred.append(pix)

x_pred = pd.concat(df_pred)
x_pred = x_pred.values
print(x_pred.shape)       # (1280000, 256) >>> (5000, 256, 256, 1)
x_pred = x_pred.reshape(5000, 256, 256, 1)
x_pred = x_pred/255
x_pred = x_pred.astype('float32')

# ================== 모델링 ==============================
def modeling() :
    # 1.
    input1 = Input(shape = (28,28,1))
    model1 = Conv2D(16,(3,3),activation='relu',padding='same')(input1)
    model1 = BatchNormalization()(model1)
    model1 = Dropout(0.3)(model1)
    model1 = Conv2D(32,(3,3),activation='relu',padding='same')(model1)
    model1 = BatchNormalization()(model1)
    model1 = Conv2D(32,(5,5),activation='relu',padding='same')(model1)
    model1 = BatchNormalization()(model1)
    model1 = Conv2D(32,(5,5),activation='relu',padding='same')(model1)
    model1 = BatchNormalization()(model1)
    model1 = Conv2D(32,(5,5),activation='relu',padding='same')(model1)
    model1 = BatchNormalization()(model1)
    model1 = MaxPooling2D(3,3)(model1)
    model1 = Dropout(0.3)(model1)
    model1 = Flatten()
    
    # 2.
    input2 = Input(shape = (256, 256, 1))
    model2 = Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same',input_shape=(256, 256, 1))(input2)
    model2 = BatchNormalization()(model2)
    model2 = AveragePooling2D(3,3)(model2)
    model2 = Dropout(0.2)(model2)
    model2 = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(model2)
    model2 = BatchNormalization()(model2) 
    model2 = Dropout(0.2)(model2)
    model2 = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(model2)
    model2 = BatchNormalization()(model2) 
    model2 = Dropout(0.2)(model2)
    model2 = Flatten()

    from tensorflow.keras.layers import concatenate, Concatenate
    merge1 = concatenate([model1, model2])

    model=Flatten()(merge1)
    model=Dense(64, activation='relu')(model)
    model=BatchNormalization()(model)
    model=Dropout(0.2)(model)

    # 1
    output1 = Dense(30)(model, activation='relu')
    output1 = Dense(26, activation='sigmoid')(output1)
    # 2
    output2 = Dense(30)(model, activation='relu')
    output2 = Dense(26, activation='sigmoid')(output2)

    models = Model(inputs = [input1, input2], outputs = [output1, output2])
    return models

re = ReduceLROnPlateau(patience=50, verbose=1, factor= 0.5)
ea = EarlyStopping(patience=100, verbose=1, mode='auto')
epochs = 1000
#KFlod대신 StratifiedKFold 써보기
#stratified 는 label 의 분포를 유지, 각 fold가 전체 데이터셋을 잘 대표한다.
skf = StratifiedKFold(n_splits=15, random_state=42, shuffle=True) #n_splits 몇 번 반복
val_loss_min = []
result = 0
nth = 0
t_d = train['letter'] # y 값 부여

for train_index, valid_index in skf.split(train2, t_d):
    x_train = train2[train_index]
    x_valid = train2[valid_index]
    y_train = t_d[train_index]
    y_valid = t_d[valid_index]
    # print(x_train.shape, x_valid.shape) #(1946, 28, 28, 1), (102, 28, 28, 1)
    # print(y_train.shape, y_valid.shape) #(1946,) (102,)

    # 실시간 데이터 증강을 사용해 배치에 대해서 모델을 학습(fit_generator에서 할 것)
    train_generator = idg.flow(x_train,y_train,batch_size=8) #훈련데이터셋을 제공할 제네레이터를 지정
    valid_generator = idg2.flow(x_valid,y_valid) # validation_data에 넣을 것
    test_generator = idg2.flow(test2,shuffle=False)  # predict(x_test)와 같은 역할
    
    model = modeling()
    mc = ModelCheckpoint('../data/modelcheckpoint/0217_1_best_mc.h5', save_best_only=True, verbose=1)
    model.compile(loss = 'binary_crossentropy', optimizer=Adam(lr=0.002,epsilon=None) ,metrics=['acc']) # y의 acc가 목적
    img_fit = model.fit_generator(train_generator,epochs=epochs, validation_data=valid_generator, callbacks=[ea,mc,re])
    
    # predict
    model.load_weights('../data/modelcheckpoint/0222_1_best_mc.h5')
    result += model.predict_generator(test_generator,verbose=True)/40 #a += b는 a= a+b
    # predict_generator 예측 결과는 클래스별 확률 벡터로 출력
    print('result:', result)

    # save val_loss
    hist = pd.DataFrame(img_fit.history)
    val_loss_min.append(hist['val_loss'].min())
    nth += 1
    print(nth, 'set complete!!') # n_splits 다 돌았는지 확인
#제출========================================
sub = pd.read_csv('./dirty_mnist_data/sample_submission.csv')
sub = result.argmax(1) # y값 index 2번째에 저장
sub.to_csv('./0222_1_result.csv',index=False)