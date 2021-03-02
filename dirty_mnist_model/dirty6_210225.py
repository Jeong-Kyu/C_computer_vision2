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
train = train.drop(['id','digit'],1) # 인덱스 있는 3개 버리기
test = test.drop(['id'],1) #인덱스 있는 것 버리기
t = range(784)
t=list(t)
A_train = train.loc[train['letter'].str.contains("A")]
A_train = pd.DataFrame(index=range(0,1), columns=[t])
A_train.insert(0,"letter",['A'],True)
A_train = A_train.fillna(0)
B_train = train.loc[train['letter'].str.contains("B")]
C_train = train.loc[train['letter'].str.contains("C")]
C_train = pd.DataFrame(index=range(0,1), columns=[t])
C_train.insert(0,"letter",['C'],True)
C_train = C_train.fillna(0)
D_train = train.loc[train['letter'].str.contains("D")]
D_train = pd.DataFrame(index=range(0,1), columns=[t])
D_train.insert(0,"letter",['D'],True)
D_train = D_train.fillna(0)
E_train = train.loc[train['letter'].str.contains("E")]
E_train = pd.DataFrame(index=range(0,1), columns=[t])
E_train.insert(0,"letter",['E'],True)
E_train = E_train.fillna(0)
F_train = train.loc[train['letter'].str.contains("F")]
F_train = pd.DataFrame(index=range(0,1), columns=[t])
F_train.insert(0,"letter",['F'],True)
F_train = F_train.fillna(0)
G_train = train.loc[train['letter'].str.contains("G")]
H_train = train.loc[train['letter'].str.contains("H")]
I_train = train.loc[train['letter'].str.contains("I")]
I_train = pd.DataFrame(index=range(0,1), columns=[t])
I_train.insert(0,"letter",['I'],True)
I_train = I_train.fillna(0)
J_train = train.loc[train['letter'].str.contains("J")]
K_train = train.loc[train['letter'].str.contains("K")]
L_train = train.loc[train['letter'].str.contains("L")]
L_train = pd.DataFrame(index=range(0,1), columns=[t])
L_train.insert(0,"letter",['L'],True)
L_train = L_train.fillna(0)
M_train = train.loc[train['letter'].str.contains("M")]
N_train = train.loc[train['letter'].str.contains("N")]
N_train = pd.DataFrame(index=range(0,1), columns=[t])
N_train.insert(0,"letter",['N'],True)
N_train = N_train.fillna(0)
O_train = train.loc[train['letter'].str.contains("O")]
P_train = train.loc[train['letter'].str.contains("P")]
P_train = pd.DataFrame(index=range(0,1), columns=[t])
P_train.insert(0,"letter",['P'],True)
P_train = P_train.fillna(0)
Q_train = train.loc[train['letter'].str.contains("Q")]
R_train = train.loc[train['letter'].str.contains("R")]
R_train = pd.DataFrame(index=range(0,1), columns=[t])
R_train.insert(0,"letter",['R'],True)
R_train = R_train.fillna(0)
S_train = train.loc[train['letter'].str.contains("S")]
T_train = train.loc[train['letter'].str.contains("T")]
T_train = pd.DataFrame(index=range(0,1), columns=[t])
T_train.insert(0,"letter",['T'],True)
T_train = T_train.fillna(0)
U_train = train.loc[train['letter'].str.contains("U")]
V_train = train.loc[train['letter'].str.contains("V")]
W_train = train.loc[train['letter'].str.contains("W")]
X_train = train.loc[train['letter'].str.contains("X")]
Y_train = train.loc[train['letter'].str.contains("Y")]
Y_train = pd.DataFrame(index=range(0,1), columns=[t])
Y_train.insert(0,"letter",['Y'],True)
Y_train = Y_train.fillna(0)
Z_train = train.loc[train['letter'].str.contains("Z")]

A_test = test.loc[test['letter'].str.contains("A")]
A_test = pd.DataFrame(index=range(0,1), columns=[t])
A_test.insert(0,"letter",['A'],True)
A_test = A_test.fillna(0)
B_test = test.loc[test['letter'].str.contains("B")]
C_test = test.loc[test['letter'].str.contains("C")]
C_test = pd.DataFrame(index=range(0,1), columns=[t])
C_test.insert(0,"letter",['C'],True)
C_test = C_test.fillna(0)
D_test = test.loc[test['letter'].str.contains("D")]
D_test = pd.DataFrame(index=range(0,1), columns=[t])
D_test.insert(0,"letter",['D'],True)
D_test = D_test.fillna(0)
E_test = test.loc[test['letter'].str.contains("E")]
E_test = pd.DataFrame(index=range(0,1), columns=[t])
E_test.insert(0,"letter",['E'],True)
E_test = E_test.fillna(0)
F_test = test.loc[test['letter'].str.contains("F")]
F_test = pd.DataFrame(index=range(0,1), columns=[t])
F_test.insert(0,"letter",['F'],True)
F_test = F_test.fillna(0)
G_test = test.loc[test['letter'].str.contains("G")]
H_test = test.loc[test['letter'].str.contains("H")]
I_test = test.loc[test['letter'].str.contains("I")]
I_test = pd.DataFrame(index=range(0,1), columns=[t])
I_test.insert(0,"letter",['I'],True)
I_test = I_test.fillna(0)
J_test = test.loc[test['letter'].str.contains("J")]
K_test = test.loc[test['letter'].str.contains("K")]
L_test = test.loc[test['letter'].str.contains("L")]
L_test = pd.DataFrame(index=range(0,1), columns=[t])
L_test.insert(0,"letter",['L'],True)
L_test = L_test.fillna(0)
M_test = test.loc[test['letter'].str.contains("M")]
N_test = test.loc[test['letter'].str.contains("N")]
N_test = pd.DataFrame(index=range(0,1), columns=[t])
N_test.insert(0,"letter",['N'],True)
N_test = N_test.fillna(0)
O_test = test.loc[test['letter'].str.contains("O")]
P_test = test.loc[test['letter'].str.contains("P")]
P_test = pd.DataFrame(index=range(0,1), columns=[t])
P_test.insert(0,"letter",['P'],True)
P_test = P_test.fillna(0)
Q_test = test.loc[test['letter'].str.contains("Q")]
R_test = test.loc[test['letter'].str.contains("R")]
R_test = pd.DataFrame(index=range(0,1), columns=[t])
R_test.insert(0,"letter",['R'],True)
R_test = R_test.fillna(0)
S_test = test.loc[test['letter'].str.contains("S")]
T_test = test.loc[test['letter'].str.contains("T")]
T_test = pd.DataFrame(index=range(0,1), columns=[t])
T_test.insert(0,"letter",['T'],True)
T_test = T_test.fillna(0)
U_test = test.loc[test['letter'].str.contains("U")]
V_test = test.loc[test['letter'].str.contains("V")]
W_test = test.loc[test['letter'].str.contains("W")]
X_test = test.loc[test['letter'].str.contains("X")]
Y_test = test.loc[test['letter'].str.contains("Y")]
Y_test = pd.DataFrame(index=range(0,1), columns=[t])
Y_test.insert(0,"letter",['Y'],True)
Y_test = Y_test.fillna(0)
Z_test = test.loc[test['letter'].str.contains("Z")]

# print(A_train.shape) 

A_train = A_train.values
B_train = B_train.values
C_train = C_train.values
D_train = D_train.values
E_train = E_train.values
F_train = F_train.values
G_train = G_train.values
H_train = H_train.values
I_train = I_train.values
J_train = J_train.values
K_train = K_train.values
L_train = L_train.values
M_train = M_train.values
N_train = N_train.values
O_train = O_train.values
P_train = P_train.values
Q_train = Q_train.values
R_train = R_train.values
S_train = S_train.values
T_train = T_train.values
U_train = U_train.values
V_train = V_train.values
W_train = W_train.values
X_train = X_train.values
Y_train = Y_train.values
Z_train = Z_train.values
A_test = A_test.values
B_test = B_test.values
C_test = C_test.values
D_test = D_test.values
E_test = E_test.values
F_test = F_test.values
G_test = G_test.values
H_test = H_test.values
I_test = I_test.values
J_test = J_test.values
K_test = K_test.values
L_test = L_test.values
M_test = M_test.values
N_test = N_test.values
O_test = O_test.values
P_test = P_test.values
Q_test = Q_test.values
R_test = R_test.values
S_test = S_test.values
T_test = T_test.values
U_test = U_test.values
V_test = V_test.values
W_test = W_test.values
X_test = X_test.values
Y_test = Y_test.values
Z_test = Z_test.values

train = np.concatenate([A_train,B_train,C_train,D_train,E_train,F_train,G_train,H_train,I_train,J_train,K_train,L_train,M_train,N_train,O_train,P_train,Q_train,R_train,S_train,T_train,U_train,V_train,W_train,X_train,Y_train,Z_train], axis =0)
test = np.concatenate([A_test,B_test,C_test,D_test,E_test,F_test,G_test,H_test,I_test,J_test,K_test,L_test,M_test,N_test,O_test,P_test,Q_test,R_test,S_test,T_test,U_test,V_test,W_test,X_test,Y_test,Z_test], axis =0)


# train2 = train.drop(['id','digit'],1) # 인덱스 있는 3개 버리기
# test2 = test.drop(['id'],1) #인덱스 있는 것 버리기
# train2 = train2.values
# test2 = test2.values

train = np.concatenate([train, test], axis =0)
# print(train.shape)


x = train[:,1:]
y = train[:,0]
x=np.array(x)
y1=np.array(y)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = OneHotEncoder()
y = encoder.fit_transform(y1.reshape(-1,1)).toarray()
r1 = np.where(x<20,0,x)
x = np.where(r1>=20,255,r1)
x = x.reshape(x.shape[0],28,28)




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
epochs = 50
#KFlod대신 StratifiedKFold 써보기
#stratified 는 label 의 분포를 유지, 각 fold가 전체 데이터셋을 잘 대표한다.
skf = StratifiedKFold(n_splits=15, random_state=42, shuffle=True) #n_splits 몇 번 반복
val_loss_min = []
result = 0
nth = 0

train_generator = idg.flow(x_train,y_train,batch_size=64) #훈련데이터셋을 제공할 제네레이터를 지정
test_generator = idg.flow(x_test,y_test) # validation_data에 넣을 것

model = modeling()
mc = ModelCheckpoint('../data/modelcheckpoint/0225_1_best_mc_2.h5', save_best_only=True, verbose=1)
model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr=0.002,epsilon=None) ,metrics=['acc']) # y의 acc가 목적
img_fit = model.fit(train_generator,batch_size=32,epochs=epochs,validation_data=test_generator, callbacks=[ea,mc,re])
model.save('C:/computervision2/dirty_mnist_model/mnist_model_2.h5')

# img_fit = model.fit_generator(train_generator,epochs=epochs, validation_data=test_generator, callbacks=[ea,mc,re])

# predict
model.load_weights('../data/modelcheckpoint/0225_1_best_mc_2.h5')
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
# sub.to_csv('./0225_2_result.csv',index=False)'''