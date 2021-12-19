# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 10:23:42 2021

@author: Gahyun
"""


import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import StratifiedKFold

# 세팅
dsrc = 'cassava-disease' #'./vision/'
nbatch = 32
nClass = 5 #3
sinput = 224 #100
epochs = 100
nfold = 10
accuracy = 0
    

# 모델 정의. 최상단 fully connected layer 제외
model = tf.keras.models.Sequential()
model.add(MobileNetV2(input_shape=(sinput,sinput,3), 
                      include_top=False, 
                      weights="imagenet"))
# Pooling + classification을 위해 최상단에 fully connected layer 추가
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(nClass, activation='softmax'))

# 데이터 생성
data_path = glob.glob('cassava-disease\\training_set/*/*')
data_label= [path1.split('\\')[-2] for path1 in data_path]
skf = StratifiedKFold(n_splits=nfold)
# preprocessing = tf.keras.applications.mobilenet_v2.preprocess_input()
train_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=85,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               zoom_range=[0.5, 1.5])
valid_gen = ImageDataGenerator(rescale=1./255)

for train_index, val_index in skf.split(data_path, data_label):
    train_data = np.array(data_path)[train_index]
    valid_data = np.array(data_path)[val_index]

    # 학습용
    train_set = train_gen.flow_from_directory(os.PathLike(train_data),
                                              target_size=(sinput,sinput),
                                              subset='training',
                                              batch_size=nbatch,
                                              class_mode='categorical')
    # 검증용
    valid_set = valid_gen.flow_from_directory(os.PathLike(valid_data),
                                              target_size=(sinput,sinput),
                                              subset='validation',
                                              batch_size=nbatch,
                                              class_mode='categorical')

    # callback 정의
    callback_ES = EarlyStopping(monitor='val_loss', 
                                min_delta=0, 
                                patience=20, 
                                verbose=1,
                                restore_best_weights=True)
    callback_MC = ModelCheckpoint('checkpoint/',
                                  monitor='val_loss',
                                  mode='min',
                                  verbose=1,
                                  save_best_only=True)
    callbacks = [callback_ES, callback_MC]
    
    # 최적화는 Adam, 손실함수는 categorical crossentropy, 평가 기준은 정확도(accuracy)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # 학습
    history = model.fit(train_set, 
                        epochs=epochs,
                        callbacks=callbacks,
                        steps_per_epoch=train_set.samples/nbatch, 
                        validation_data=valid_set,
                        validation_steps=train_set.samples/nbatch, 
                        verbose = 1)
    results = model.evaluate(valid_set)
    results = dict(zip(model.metrics_names, results))
    accuracy += results['acc'] / nfold

# 평가
# pred = model.predict(test_set)
# pred_val = np.argmax(pred, axis=1)
# pred_file = test_set.filenames
# accuracy = model.evaluate_generator(test_set)
# print('accuracy is {:.3f}%.'.format(accuracy[1]*100))
# print('accuracy[0] : {:.6f}.'.format(accuracy[0]))


