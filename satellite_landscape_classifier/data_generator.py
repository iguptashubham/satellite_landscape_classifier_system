from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def TrainVal():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=(0.1, 0.4),
        shear_range=0.4,
        zoom_range=0.5,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=True
    )

    train_ds = datagen.flow_from_directory(
        r'satellite_landscape_classification\data\train',
        target_size=(255, 255),
        class_mode='categorical'
    )

    val_ds = datagen.flow_from_directory(
        r'satellite_landscape_classification\data\val',
        target_size=(255, 255),
        class_mode='categorical'
    )
    
    return train_ds, val_ds

# "train_ds, _ = TrainVal()

# # Get class names from the class indices dictionary
# class_names = list(train_ds.class_indices.keys())
# print(class_names)
