"""
Trains a data model with MOTOR-TALK forum posts (from the “data” directory).
"""

import time

from utils import reduce_logging_output, load_dataset, create, train

# ------
# Config
# ------

data_base_path = "data"
text_embedding_module = "https://tfhub.dev/google/nnlm-de-dim128/1"
training_steps = 1000
model_dir = "models/checkpoint"

# -----------
# Preparation
# -----------

start_time = time.time()
reduce_logging_output()

# ---------
# Load Data
# ---------

print("*** LOADING ***")

training_data = load_dataset(data_base_path)

print("dataset loading DONE:")
print(training_data.head())

# ---------------
# Train Estimator
# ---------------

print("*** CREATING ***")
estimator = create(text_embedding_module, model_dir)

print("*** TRAINING ***")
train(estimator, training_data, training_steps)
print("estimator training DONE")

print("Elapsed time: {0:.0f} sec".format(time.time() - start_time))
