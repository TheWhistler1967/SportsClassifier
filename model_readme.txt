Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 34, 34, 64)        36928     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 17, 17, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 17, 17, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 18496)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               2367616   
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 516       
=================================================================
Total params: 2,424,452
Trainable params: 2,424,452
Non-trainable params: 0
_________________________________________________________________
Found 116 images belonging to 4 classes.
Found 12 images belonging to 4 classes.
----------------------------------------------------------------
Accuracy: 95% on validation/testing. Trained over 2000 epochs, (which was far too many).

This model is overfitting: 100% train vs around 90% on validation. But this is possibly due to small data and single 'unusual' instances eg. uncommon ball colours, shapes, and angles - which only exist once. So if the validation set is using these, then the model has never seen anything like it before.

To mitigate this: could try b&w images, but this is unlikely to make much difference due to the number of features, and there is actually a surprising variety of colours so I would think it unlikely that the model would rely too heavily on colour. Could create a custom validation set that is more fair on the model. But most of all, I need much more data.