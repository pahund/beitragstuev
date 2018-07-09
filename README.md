# MOTOR-TALK Beitrags-TÜV

Scripts for classifying automotive forum posts with machine learning, using Google's 
[TensorFlow](https://www.tensorflow.org) framework.

Allows us to rate the quality of forum posts.

Available scripts:

* `train-model.py` – creates an classifier model and trains it with posts from MOTOR-TALK (run this first)
* `classify-text.py` – classifies some example texts (run after training the model)
* `save-model.py` – saves the model for later use (run after training the model)
* `prepare-data.sh` – helper script to prepare MT forum posts from the database for processing

The `data` directory contains texts from the MT database that have already been prepared:

* `data/pos` – Posts from the FAQs, to be classified as *POSITIVE* (i.e. good quality posts)
* `data/neg` – Posts that were deleted by the admins, to be classified as *NEGATIVE* (i.e. low quality posts)
