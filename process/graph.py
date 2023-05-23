from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
from tqdm import tqdm

sys.path.append('..')
from config import *

if __name__ == '__main__':
    print(TRAIN_CSV_DIR)