# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 16:26:30 2023

@author: 1770096
"""

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import tensorflow_hub as hub

import tensorflow_datasets as tfds

from tensorflow.keras import layers

import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)


#Flower data Set 다운로드,training_set 및 validation_set으로 분할

(training_set, validation_set), dataset_info = tfds.load(

    'tf_flowers',

    split=['train[:70%]', 'train[70%:]'],

    with_info=True,

    as_supervised=True,

)



#data set의 클래스 수 출력 및 training set, validation set에 있는 이미지 수 계산
num_classes = dataset_info.features['label'].num_classes

 
num_training_examples = 0

num_validation_examples = 0


for example in training_set:

  num_training_examples += 1


for example in validation_set:

  num_validation_examples += 1


print('Total Number of Classes: {}'.format(num_classes))

print('Total Number of Training Images: {}'.format(num_training_examples))

print('Total Number of Validation Images: {} \n'.format(num_validation_examples))




#Flowers 데이터 집합의 이미지 크기 확인(각각 다름)
for i, example in enumerate(training_set.take(5)):

  print('Image {} shape: {} label: {}'.format(i+1, example[0].shape, example[1]))
  
  
  
  
  
 #MobileNet V2에 쓰이는 해상도로 모든 이미지 재포맷(전처리과정): 이미지크기 224*224, 255로 나누어 normalization
  IMAGE_RES = 224


def format_image(image, label):

  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0

  return image, label


BATCH_SIZE = 32


train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)


validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)



##Tensorflow Hub 사용하여 전이학습 수행

#URL을 통해 MobileNet v2 가져오기,feature_extractor(input_shape 매개 변수를 가진 KerasLayer)를 생성.
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL,

                                   input_shape=(IMAGE_RES, IMAGE_RES, 3))




#특징추출기 계층 변수 동결(훈련은 분류기 계층 변수만 변경하도록..)
feature_extractor.trainable = False




# tf.keras.Sequential 모델을 만들고, 사전 학습 모델과 새 분류 계층을 추가
model = tf.keras.Sequential([

  feature_extractor,

  layers.Dense(num_classes)

])

model.summary()




#모델 훈련(6회)
model.compile(

  optimizer='adam',

  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

  metrics=['accuracy'])


EPOCHS = 6


history = model.fit(train_batches,

                    epochs=EPOCHS,

                    validation_data=validation_batches)





#Training, Validation의 Accuracy, Loss그래프 표시
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

 
loss = history.history['loss']

val_loss = history.history['val_loss']


epochs_range = range(EPOCHS)

 
plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')
 

plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()




#데이터 집합 정보에서 레이블 이름을 가져와 NumPy 배열로 변환, 배열 인쇄
class_names = np.array(dataset_info.features['label'].names)
 
print(class_names)





#최상의 예측 인덱스를 클래스 이름으로 변환
image_batch, label_batch = next(iter(train_batches))
 

image_batch = image_batch.numpy()

label_batch = label_batch.numpy()

 
predicted_batch = model.predict(image_batch)

predicted_batch = tf.squeeze(predicted_batch).numpy()
 

predicted_ids = np.argmax(predicted_batch, axis=-1)

predicted_class_names = class_names[predicted_ids]


print(predicted_class_names)





#실제레이블과 예측레이블 인덱스 인쇄
print("Labels:           ", label_batch)

print("Predicted labels: ", predicted_ids)





#Plot Model Predictions
plt.figure(figsize=(10,9))

for n in range(30):

  plt.subplot(6,5,n+1)

  plt.subplots_adjust(hspace = 0.3)

  plt.imshow(image_batch[n])

  color = "blue" if predicted_ids[n] == label_batch[n] else "red"

  plt.title(predicted_class_names[n].title(), color=color)

  plt.axis('off')

_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")