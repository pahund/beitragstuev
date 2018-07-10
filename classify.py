#!/usr/bin/env python3
"""
Restores the data model created by running train-model.py and classifies text from stdin.
"""

import sys
import time
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from config import text_embedding_module, model_dir
from utils import reduce_logging_output, create, print_analysis

start_time = time.time()
reduce_logging_output()

estimator = create(text_embedding_module, model_dir)
for line in sys.stdin.readlines():
    print_analysis(line.strip(), estimator)

print("Elapsed time: {0:.0f} sec".format(time.time() - start_time))
