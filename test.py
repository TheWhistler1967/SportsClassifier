#!/usr/bin/python

import os
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image as prepro_image
from imutils import paths
from sklearn.preprocessing import LabelBinarizer


def load_img(test_data_dir, image_size = (150, 150)):
    images_data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(test_data_dir)))
    for imagePath in imagePaths:
        image = prepro_image.load_img(imagePath, target_size=image_size)
        image = img_to_array(image)
        images_data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    return images_data, sorted(labels)


def convert(images, labels):
    x_test = np.array(images, dtype="float")/255.0
    y_test = np.array(labels)
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)
    return x_test, y_test


def evaluate(X_test, y_test):
    batch_size = 16
    model = load_model('model.h5')
    return model.evaluate(X_test, y_test, batch_size, verbose=1)


if __name__ == '__main__':
    test_data_dir = 'dataset/test'
    image_size = (150, 150)

    # Load
    images, labels = load_img(test_data_dir, image_size)

    # Convert images to numpy arrays, and binarise
    X_test, y_test = convert(images, labels)

    # Test it
    loss, accuracy = evaluate(X_test, y_test)
    print("loss={}, accuracy={}".format(loss, accuracy))