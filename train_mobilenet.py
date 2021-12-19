# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 16:41:00 2021

@author: Gahyun
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 세팅
dsrc = './vision/'
nbatch = 16
nClass = 3
sinput = 224 #100
epochs = 200
    

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
image_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=85,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               validation_split=0.2)

# 학습용
train_set = image_gen.flow_from_directory(os.path.join(dsrc, 'training_set'),
                                          target_size=(sinput,sinput),
                                          subset='training',
                                          batch_size=nbatch,
                                          class_mode='categorical')
# 검증용
valid_set = image_gen.flow_from_directory(os.path.join(dsrc, 'training_set'),
                                          target_size=(sinput,sinput),
                                          subset='validation',
                                          batch_size=nbatch,
                                          class_mode='categorical')
# 평가용
test_gen = ImageDataGenerator(rescale=1./255)
test_set = test_gen.flow_from_directory(os.path.join(dsrc, 'test_set'),
                                        batch_size=1,
                                        shuffle=False,
                                        target_size=(sinput,sinput))

# callback 정의
# callback_ES = EarlyStopping(monitor='val_loss', 
#                             min_delta=0, 
#                             patience=50, 
#                             verbose=1,
#                             restore_best_weights=True)
# callback_MC = ModelCheckpoint('checkpoint/',
#                               monitor='val_loss',
#                               mode='min',
#                               verbose=1,
#                               save_best_only=True)
# callbacks = [callback_ES, callback_MC]

# 최적화는 Adam, 손실함수는 categorical crossentropy, 평가 기준은 정확도(accuracy)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 
nfile = np.array([216, 150, 839])
weights = [4., 5., 2.]#nfile[-1::-1] / sum(nfile)
class_weight = {0: weights[0], 1: weights[1], 2: weights[2]}
# 학습
history = model.fit(train_set, 
                    epochs=epochs,
                    # callbacks=callback_MC,
                    class_weight=class_weight,
                    steps_per_epoch=train_set.samples/nbatch, 
                    validation_data=valid_set,
                    validation_steps=train_set.samples/nbatch, 
                    verbose = 1)

# 평가
pred = model.predict(test_set)
pred_idx = np.argmax(pred, axis=1)
pred_file = test_set.filenames
accuracy = model.evaluate(test_set)
print('accuracy is {:.3f}%.'.format(accuracy[1]*100))
print('accuracy[0] : {:.6f}.'.format(accuracy[0]))


# # 제출
submit = pd.DataFrame(list(zip(pred_file, pred_idx)), columns=['Name', 'pred'])
submit.to_csv('submission'+str(epochs)+'.csv', index=False)

