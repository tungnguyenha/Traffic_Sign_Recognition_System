import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
import pickle
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

data = []
labels = []
classes = 43
imageDimesions = (32,32,3)
cur_path = os.getcwd()

#Truy xuất hình ảnh và nhãn của chúng
for i in range(classes):
    path = os.path.join(cur_path,'Train',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '/'+ a)
            image = image.resize((32,32))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

#Chuyển đổi danh sach thành mảng numpy
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

#Tách tập dữ liệu huấn luyện và thử nghiệm
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # type: ignore

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img
X_train = np.array(list(map(preprocessing,X_train))) # TO IRETATE AND PREPROCESS ALL IMAGES
X_test = np.array(list(map(preprocessing,X_test)))
print(X_train.shape)


#Chuyển đổi nhãn thành một mã hoá
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#Xây dựng mô hình
cnn_model = Sequential()
cnn_model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(imageDimesions[0],imageDimesions[1],1)))
cnn_model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
cnn_model.add(MaxPool2D(pool_size=(2,2)))
cnn_model.add(Dropout(rate=0.25))
cnn_model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
cnn_model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
cnn_model.add(MaxPool2D(pool_size=(2,2)))
cnn_model.add(Dropout(rate=0.25))
cnn_model.add(Flatten())
cnn_model.add(Dense(256, activation='relu'))
cnn_model.add(Dropout(rate=0.5))
cnn_model.add(Dense(43, activation='softmax'))

#Tổng hợp mô hình
cnn_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

epochs = 15
history = cnn_model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

cnn_model.save('traffic_model.h5')
cnn_model.save_weights("weights_model.h5")