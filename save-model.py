#!/usr/bin/env python3
"""
Restores the data model created by running train-model.py and saves it for later use.
"""

import sys
import time
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from config import text_embedding_module, save_dir, model_dir
from utils import reduce_logging_output, create, save

start_time = time.time()
reduce_logging_output()
estimator = create(text_embedding_module, model_dir)

save(estimator=estimator, text_embedding_module=text_embedding_module, export_dir_base=save_dir)

print("Elapsed time: {0:.0f} sec".format(time.time() - start_time))
