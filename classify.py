#!/usr/bin/env python3

import sys
import time

from config import text_embedding_module, model_dir
from utils import reduce_logging_output, create, print_analysis

start_time = time.time()
reduce_logging_output()

estimator = create(text_embedding_module, model_dir)
for line in sys.stdin.readlines():
    print_analysis(line, estimator)

print("Elapsed time: {0:.0f} sec".format(time.time() - start_time))
