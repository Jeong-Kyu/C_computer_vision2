import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator # 이미지데이터 늘리는 작업
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential,Model
from keras.layers import *
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, MaxPooling1D, Conv1D, AveragePooling2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('C:/computervision2_data/dirty_mnist_data/mnist_data/train.csv')
test = pd.read_csv('C:/computervision2_data/dirty_mnist_data/mnist_data/test.csv')
submission = pd.read_csv('C:/computervision2_data/dirty_mnist_data/mnist_data/submission.csv')

train2 = train.drop(['id','digit'],1) # 인덱스 있는 3개 버리기
test2 = test.drop(['id'],1) #인덱스 있는 것 버리기
train2 = train2.values
test2 = test2.values

train = np.concatenate([train2, test2], axis =0)
print(train.shape)

x = train[:,1:]
y = train[:,0]
x=np.array(x)
y1=np.array(y)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(y1.reshape(-1,1)).toarray()
print(y.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66 )
print(type(x_test))
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_train = x_train/255
x_train = x_train.astype('float32')
x_test = x_test/255
x_test = x_test.astype('float32')

idg = ImageDataGenerator( 
                                  rotation_range=15,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.5,
                                  zoom_range=[0.8, 2.0],
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='nearest')
    # height_shift_range=(-1,1),width_shift_range=(-1,1)) # 이미지 카테고리화(4차원만 가능)
idg2 = ImageDataGenerator()

def modeling() :
    model = Sequential()
    model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3,3)))
    model.add(Dropout(0.3))
    
    model.add(Flatten()) #2차원
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(26,activation='softmax')) # softmax는 'categorical_crossentropy' 짝꿍
    return model
    
re = ReduceLROnPlateau(patience=20, verbose=1, factor= 0.5)
ea = EarlyStopping(patience=50, verbose=1, mode='auto')
epochs = 3000
#KFlod대신 StratifiedKFold 써보기
#stratified 는 label 의 분포를 유지, 각 fold가 전체 데이터셋을 잘 대표한다.
skf = StratifiedKFold(n_splits=15, random_state=42, shuffle=True) #n_splits 몇 번 반복
val_loss_min = []
result = 0
nth = 0

train_generator = idg.flow(x_train,y_train,batch_size=64) #훈련데이터셋을 제공할 제네레이터를 지정
test_generator = idg.flow(x_test,y_test) # validation_data에 넣을 것

model = modeling()
mc = ModelCheckpoint('../data/modelcheckpoint/0225_1_best_mc_4.h5', save_best_only=True, verbose=1)
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None) ,metrics=['acc']) # y의 acc가 목적
img_fit = model.fit(train_generator,batch_size=32,epochs=epochs,validation_data=test_generator, callbacks=[ea,mc,re])
model.save('C:/computervision2/dirty_mnist_model/mnist_model_1.h5')

# img_fit = model.fit_generator(train_generator,epochs=epochs, validation_data=test_generator, callbacks=[ea,mc,re])

# predict
model.load_weights('../data/modelcheckpoint/0225_1_best_mc_4.h5')
result += model.predict(test_generator,verbose=True) #a += b는 a= a+b
# predict_generator 예측 결과는 클래스별 확률 벡터로 출력
print('result:', result)

# save val_loss
hist = pd.DataFrame(img_fit.history)
val_loss_min.append(hist['val_loss'].min())
nth += 1
print(nth, 'set complete!!') # n_splits 다 돌았는지 확인
#제출========================================
# sub = pd.read_csv('C:/computervision2/dirty_mnist_data/sample_submission.csv')
# sub['digit'] = result.argmax(1) # y값 index 2번째에 저장
# sub.to_csv('./0225_2_result.csv',index=False)