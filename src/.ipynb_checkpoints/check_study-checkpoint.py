import os
import scipy
import pickle
import optuna
import logging
import argparse
import warnings
import joblib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tmu.data import MNIST
from tmu.tools import BenchmarkTimer
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

from train_tm import load_data, load_orig_data
parser = argparse.ArgumentParser()
parser.add_argument("--data_type", default='orig', type=str)
parser.add_argument("--upscale", default=False, type=bool)
parser.add_argument("--downscale", default=False, type=bool)
parser.add_argument("--tm_type", default='vanilla', type=str)
parser.add_argument("--num_classes", default=3, type=int)
args = parser.parse_args()
study = joblib.load(f"hpsearch/hp_{args.tm_type}_{args.upscale}_{args.downscale}_{args.data_type}_{args.num_classes}.pkl")
print(study.best_params)
print(study.best_value)
save_path = f"plots/hp_{args.tm_type}_{args.upscale}_{args.downscale}_{args.data_type}_{args.num_classes}"
if not os.path.exists(save_path):
    os.mkdir(save_path)
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image(f"{save_path}/optimization_history.png")  

fig = optuna.visualization.plot_parallel_coordinate(study)
fig.write_image(f"{save_path}/parallel_coordinate.png")

fig = optuna.visualization.plot_param_importances(study)
fig.write_image(f"{save_path}/param_importances.png")











