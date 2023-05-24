import sys
import os
import pandas as pd
from glob import glob
from collections import Counter

sys.path.append('..')
from graph import *
from ocr import *

if __name__ == '__main__':
    print(os.getcwd())