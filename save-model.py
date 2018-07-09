"""
Trains a data model with MOTOR-TALK forum posts (from the “data” directory) and saves it for later use.
"""

import time

from utils import reduce_logging_output, load_dataset, create, save

# ------
# Config
# ------

text_embedding_module = "https://tfhub.dev/google/nnlm-de-dim128/1"
model_dir = "models/checkpoint"
save_dir = "models/saved"

# -----------
# Preparation
# -----------

start_time = time.time()
reduce_logging_output()
estimator = create(text_embedding_module, model_dir)

# ----
# Save
# ----

print("*** SAVING ***")
save(estimator=estimator, text_embedding_module=text_embedding_module, export_dir_base=save_dir)
print("Elapsed time: {0:.0f} sec".format(time.time() - start_time))
