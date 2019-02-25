#!/usr/bin/python
'''
This script will monitor the AutoNamingFolder, and when it sees a new ball image it will use a previously trained model to classify it
into either 'Basketball, Rugby', 'soccer' or 'Tennis'. Model was trained on clean images (no noise). Ouputs class_prob.
'''

import os
import numpy as np
import tensorflow as tf
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from keras.models import load_model
from skimage import transform
from PIL import Image

folder_name = 'AutoNamingFolder/'
count = 0

img_width, img_height = 150, 150
model = load_model('model.h5')
graph = tf.get_default_graph()
labels = ("Basketball", "Rugby", "Soccer", "Tennis")


class FileHandler(PatternMatchingEventHandler):
    def on_created(self, event):
        global count
        # Load image, normalise, rescale, and convert to np array
        np_image = Image.open(event.src_path)
        np_image = np.array(np_image).astype('float32') / 255
        np_image = transform.resize(np_image, (150, 150, 3))
        np_image = np.expand_dims(np_image, axis=0)

        with graph.as_default():  # Default graph loaded at top
            y = model.predict(np_image)
            y_prob = model.predict_proba(np_image)

        print(y_prob)
        class_lab = y.argmax(axis=-1)
        os.rename(event.src_path, '{}{}_{}.{}'.format(folder_name, count, labels[class_lab[0]], 'jpg'))
        count += 1


if __name__ == '__main__':
    count = 0
    observer = Observer()
    observer.schedule(FileHandler(), path=folder_name)
    observer.start()

    print("Watching")  # Wait for this before dragging images into AutoNamingFolder
    observer.join()
