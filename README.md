# MOTOR-TALK Beitrags-TÜV

Scripts for classifying automotive forum posts with machine learning, using Google's 
[TensorFlow](https://www.tensorflow.org) framework.

Allows us to rate the quality of forum posts.

## Prerequisites

You need to have Python 3.6 installed (**Warning:** it seems that TensorFlow does *not* work with Python 3.7),
along with the following packages:

* tensorflow
* tensorflow_hub 
* pandas 

Command for installing the packages:

```
pip3 install tensorflow tensorflow_hub pandas
```

## Available Scripts

* `train-model.py` – creates an classifier model and trains it with posts from MOTOR-TALK (run this first)
* `classify.py` – classifies text from stdin (see usage example below)
* `classify-examples.py` – classifies some example texts (run after training the model)
* `save-model.py` – saves the model for later use (run after training the model)

## Utilities

* `prepare-data.js` – helper script to prepare MT forum posts from the database for processing 
  (requires Node.js)

## Data

The `data` directory contains texts from the MT database that have already been prepared:

* `data/pos` – Posts from the FAQs, to be classified as *POSITIVE* (i.e. good quality posts)
* `data/neg` – Posts that were deleted by the admins, to be classified as *NEGATIVE* (i.e. low quality posts)

This data is used by the `train-model.py` script.

## Classifying Text from the Command Line

Before everything else, you need to train the model using the text files in the `data` directory, with this
command:

```
./train-model.py
```

Be aware that depending on the processing power of your 'puter, this may take quite some time (~5 minutes on 
a late 2016 model MacBook Pro). The model takes up about 500 MB hard disk space, so make sure that is available.

**Caveat for colleagues from eBay:** for some reason, the *.dev* top level domain is blocked in our corp and
guest networks, this results in a network error when running the training script, because the word embedding
files required for classification can't be downloaded from *https://tfhub.dev/google/nnlm-de-dim128/1*.
You'll have to resort to running the script at home or using mobile phone tethering.

After having created and trained the model, you can use the `classify.py` script to classify text from the 
command line, for example like this:

```
echo "Das Paket „S Line Competition“ umfasst unter anderem optische Details, eine neue Farbe (Turboblau), 19-Zöller und LED-Lampen." | ./classify.py
```

This will give you a classification *positive* (i.e. it's a good post), with a score of approximately 90%.

Using a text file:

```
cat data/neg/post_1651353.txt | ./classify.py
```

(classification *negative*, 5%)
