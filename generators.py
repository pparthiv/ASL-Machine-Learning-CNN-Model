import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_val_generators(training_images, training_labels, validation_images, validation_labels):
    training_images = np.expand_dims(training_images, axis = 3)
    validation_images = np.expand_dims(validation_images, axis = 3)

    train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')
    train_generator = train_datagen.flow(x = training_images, y = training_labels, batch_size = 32)

    validation_datagen = ImageDataGenerator(rescale = 1./255)
    validation_generator = validation_datagen.flow(x = validation_images, y = validation_labels, batch_size = 32) 

    return train_generator, validation_generator