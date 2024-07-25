import matplotlib.pyplot as plt
import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

fig_obj = pickle.load(open(args.filename, 'rb'))
plt.ioff()
plt.show()