#!/usr/bin/env python3
"""
Trains a data model with MOTOR-TALK forum posts (from the “data” directory).
"""

import sys
import time
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from config import data_dir, text_embedding_module, training_steps, model_dir
from utils import reduce_logging_output, load_dataset, create, train

start_time = time.time()
reduce_logging_output()

print("*** LOADING ***")
training_data = load_dataset(data_dir)
print("dataset loading DONE:")
print(training_data.head())

print("*** TRAINING ***")
estimator = create(text_embedding_module, model_dir)
train(estimator, training_data, training_steps)
print("estimator training DONE")

print("Elapsed time: {0:.0f} sec".format(time.time() - start_time))
