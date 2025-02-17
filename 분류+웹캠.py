# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 23:02:40 2023

@author: 1770096
"""

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:\\Users\\1770096\\Desktop\\hj\\keras_Model.h5", compile=False)

# Load the labels
class_names = open("C:\\Users\\1770096\\Desktop\\hj\\labels.txt", encoding = "UTF8").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("C:\\Users\\1770096\\Desktop\\hj\\ㅂㅂ.jpg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)

##웹캠으로 연결##
import cv2
import tensorflow.keras
import numpy as np
import requests
from bs4 import BeautifulSoup

## 이미지 전처리
def preprocessing(frame):
    # 사이즈 조정
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    # 이미지 정규화
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
    
    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    
    return frame_reshaped

## 학습된 모델 불러오기
model_filename = 'C:\\Users\\1770096\\Desktop\\hj\\keras_model.h5'
model = tensorflow.keras.models.load_model(model_filename, compile=False)

# 카메라 캡쳐 객체, 0=내장 카메라
capture = cv2.VideoCapture(0)           #ipcam사용시 여기의 0을 바꿔주세요

# 캡쳐 프레임 사이즈 조절
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)


while True:
    
    ret, frame = capture.read()

    frame_fliped = cv2.flip(frame, 1)
    
    if cv2.waitKey(200) > 0: 
        break
    
    preprocessed = preprocessing(frame_fliped)
    
    prediction = model.predict(preprocessed)
    
    if prediction[0,1] < prediction[0,0] and prediction[0,2] < prediction[0,0]:
        cv2.putText(frame_fliped, 'Bossam', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        
    elif prediction[0,0] < prediction[0,1] and prediction[0,2] < prediction[0,1]:
        cv2.putText(frame_fliped, "Bibimbab", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        
    elif prediction[0,0] < prediction[0,2] and prediction[0,1] < prediction[0,2]:
        cv2.putText(frame_fliped, 'Bread', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        
    cv2.imshow("VideoFrame", frame_fliped)
capture.release() 

cv2.destroyAllWindows()
