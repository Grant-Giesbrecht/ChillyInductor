import matplotlib.pyplot as plt
import subprocess

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

cmd = f"pythonw .\plot_pklfig_blocking.py \"{args.filename}\""

# print(f"Calling: {cmd}")
subprocess.Popen(cmd)