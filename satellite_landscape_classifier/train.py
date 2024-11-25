import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
import time

def trainmodel(model,train,val):
    earlystopper = EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
    Adam_opt = Adam()
    model.compile(optimizer=Adam_opt,loss='categorical_crossentropy', metrics = ['accuracy'])
    traininginfo = model.fit_generator(train,epochs=108,validation_data=(val), callbacks = [earlystopper])
    return traininginfo


def savemodel(model):
    # Create a directory with the current timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    dir_name = os.path.join('satellite_landscape_classification\models',f'SLCS_model_{timestamp}')
    os.makedirs(dir_name, exist_ok=True)
    
    # Define the paths for the model and weights
    model_path = os.path.join(dir_name, 'SLCS.h5')
    weights_path = os.path.join(dir_name, 'SLCS_wg.weights.h5')
    
    # Save the model and weights
    model.save(model_path)
    model.save_weights(weights_path)
    
    print(f"Model and weights saved in directory: {dir_name}")
