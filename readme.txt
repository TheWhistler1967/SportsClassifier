train.py
	Training code.

test.py
	Testing code - though this is pretty useless because the selected instances are far too easy for the model to catagorise.

gpu_test.py
	Simple test to confirm the TF-GPU is working.

renamer.py
	Run this script and wait until output 'Watching'.You can then move images into 'AutoNamingFolder' and they will be predicted and named accordingly. eg. you can drag images from google images. 
Model can currently only identify:
			rugby ball
			soccer ball
			tennis ball
			basketball

test.py and renamer.py both set to use a savepoint model: 'model.h5', which was about 92% on val and 95% on test. final_model was at the end of 2000 epochs, which was 86% on valid.

But with such a small number on my valid set, it is hard to know which of these is actually performing better without setting up a full largescale test.