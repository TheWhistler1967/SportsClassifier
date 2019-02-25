#!/usr/bin/python

from keras.models import Sequential
from keras.layers import MaxPooling2D, Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import time


# image dimensions
img_width, img_height = 150, 150


def construct_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


def train_model(model):
    batch_size = 4
    epochs = 2000
    train_dir = 'dataset/train'
    #valid_dir = 'dataset/valid'

    image_datagen = ImageDataGenerator(rescale=1 / 255,
                                       rotation_range=40,
                                       horizontal_flip=True,
                                       validation_split=0.1,
                                       height_shift_range=0.2,
                                       width_shift_range=0.2,
                                       zoom_range=0.2,
                                       shear_range=0.2)

    training_set = image_datagen.flow_from_directory(train_dir,
                                                     target_size=(img_width, img_height),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     subset='training')

    valid_set = image_datagen.flow_from_directory(train_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  subset='validation')

    model.fit_generator(training_set,
                        epochs=epochs,
                        validation_data=valid_set,
                        validation_steps=valid_set.samples,
                        steps_per_epoch=training_set.samples/batch_size,
                        shuffle=True,
                        callbacks=[ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)])
    return model


if __name__ == '__main__':
    start_time = time.time()
    model = construct_model()
    model = train_model(model)
    model = model.save(filepath='final.h5')
    print("--- %s seconds ---" % (time.time() - start_time))
