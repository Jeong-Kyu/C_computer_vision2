### 제 2회 컴퓨터 비전 학습 경진대회
+ 월간 데이콘 12 / 컴퓨터 비젼 / multi-label classification
+ 2021.01.27 ~ 2021.03.01
+ DATA
  #### 1. dirty_mnist_2nd.zip (3.05 GB)
    + (256,256)크기의 .png 파일 50,000 개
    + 월간데이콘 7의 알파벳 데이터를 합성한 이미지
    + (256, 256)크기의 이미지 속에 10 ~ 14개의 글자(알파벳 a – Z, 중복 제외)가 배치
  #### 2. test_dirty_mnist_2nd.zip (313 MB)
    + (256, 256)크기의 .png 파일 5,000 개
    + dirty_mnist와 유사한 형태로 구성
  #### 3. mnist_data.zip (11.7 MB)
    + 월간데이콘 7 컴퓨터 비전 학습 경진대회에서 사용되었던 데이터
    + train.csv, test.csv, sample_submission.csv
  #### 4. dirty_mnist_2nd_answer.csv(2.80 MB)
    + dirty_mnist에 존재하는 5만장의 이미지에 대한 정답 파일
    + 알파벳이 존재할 경우 1, 존재하지 않을 경우 0로 채움
  #### 5. sample_submission.csv(288 KB)
  + 출처: https://www.nist.gov/itl/products-and-services/emnist-dataset
+ Model
  + ImageDataGenerator
  + resnet
