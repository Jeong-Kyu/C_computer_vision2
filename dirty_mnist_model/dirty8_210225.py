import pandas as pd
import cv2
import numpy as np
import os
 
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
sub = pd.read_csv('C:/computervision2_data/dirty_mnist_data/sample_submission.csv')
for c in range(50000):   
    print(c)
    large = cv2.imread('C:/computervision2_data/dirty_mnist_data/clean_dirty/train/'+str(c).zfill(5)+'.png')
    small = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    print(connected.shape)
    print(large.shape)

    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    a = []
    
    # createFolder('C:/computervision2/dirty_mnist_data/select/'+str(c).zfill(5))

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        cropped = large[y:y+h, x:x+w]
        face = cv2.resize(cropped, (28,28))
        # file_name_path = 'C:/computervision2/dirty_mnist_data/select/'+str(c).zfill(5)+'/'+str(idx)+'.jpg'
        # cv2.imwrite(file_name_path, face)
        # cv2.imshow('rects', face)
        # cv2.waitKey()

        from tensorflow.keras.models import Sequential, load_model
        face = face.reshape(28,28,3)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.reshape(1,28,28,1)
        model = load_model('C:/computervision2/dirty_mnist_model/mnist_model_1.h5')
        result = model.predict(face)
        a = np.where(result==1)[1]
        print (a)
        if not a:
            continue
        else:
            sub.loc[idx][a] = 1 # y값 index 2번째에 저장
            sub.to_csv('./0222_1_result.csv',index=False)
        
        # if r > 0.45 and w > 8 and h > 8:
        #     cv2.rectangle(large, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
        # show image with contours rect
print("끝")
