"""
Trains a data model with MOTOR-TALK forum posts (from the “data” directory) and saves it for later use.
"""

import time

from utils import reduce_logging_output, load_dataset, create_and_train, save

# ------
# Config
# ------

data_base_path = "data"
text_embedding_module = "https://tfhub.dev/google/nnlm-de-dim128/1"
training_steps = 10

# ---------
# Load Data
# ---------

start_time = time.time()
print("*** LOADING ***")

reduce_logging_output()

training_data = load_dataset(data_base_path)

# ---------------
# Train Estimator
# ---------------

print("*** TRAINING ***")
estimator = create_and_train(training_data, text_embedding_module, training_steps)

# ----
# Save
# ----

print("*** SAVING ***")
save(estimator=estimator, text_embedding_module=text_embedding_module, export_dir_base="model")
print("Elapsed time: {0:.0f} sec".format(time.time() - start_time))
