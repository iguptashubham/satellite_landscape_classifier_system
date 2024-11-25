import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def BuildModel():
    v2_conv = MobileNetV2(
        include_top=False,
        weights = 'imagenet',
        input_shape = (255,255,3),
        classes = 4
    )
    v2_conv.trainable=False #freezing layers
    
    model = Sequential()
    model.add(v2_conv)
    model.add(Flatten())
    model.add(Dense(1028,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(504,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(100,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(4,activation='softmax'))
    
    model.summary()
    return model
    
