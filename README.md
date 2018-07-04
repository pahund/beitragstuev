# MOTOR-TALK Beitrags-TÜV

Scripts for classifying automotive forum posts with machine learning, using Google's 
[TensorFlow](https://www.tensorflow.org) framework.

Allows us to rate the quality of forum posts.

Available scripts:

* `classify-text.py` – Trains a data model with MT forum posts and classifies some example texts
* `save-model.py` – Trains a data model and saves it for later use
* `prepare-data.sh` – helper script to prepare MT forum posts from the database for processing

The `data` directory contains texts from the MT database that have already been prepared:

* `data/pos` – Posts from the FAQs, to be classified as *POSITIVE* (i.e. good quality posts)
* `data/neg` – Posts that were deleted by the admins, to be classified as *NEGATIVE* (i.e. low quality posts)
