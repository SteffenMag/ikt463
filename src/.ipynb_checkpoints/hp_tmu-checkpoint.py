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
import plotly.io as pio

from tmu.data import MNIST
from tmu.tools import BenchmarkTimer
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

from train_tm import load_data, load_orig_data


_LOGGER = logging.getLogger(__name__)

def metrics():
    return dict(
        accuracy=[],
        train_time=[],
        test_time=[]
    )

def metrics_coalesced():
    return dict(
        number_of_positive_clauses=[],
        accuracy=[],
        number_of_includes=[],
        train_time=[],
        test_time=[]
    )

def main(s, n_clauses, threshold, max_included_literals, weighted_clauses, platform, epochs):
    experiment_results = metrics()
    
    '''train_x = np.load('tm_data/train_x.npy')
    train_y = np.load('tm_data/train_y.npy')
    test_x = np.load('tm_data/test_x.npy')
    test_y = np.load('tm_data/test_y.npy')'''

    #train_x, train_y, test_x, test_y = load_orig_data()

    tm = TMClassifier(
        type_iii_feedback=False,
        number_of_clauses=n_clauses,
        T=threshold,
        s=s,
        max_included_literals=max_included_literals,
        platform=platform,
        weighted_clauses=weighted_clauses
    )

    #_LOGGER.info(f"Running {TMClassifier} for {epochs}")
    for epoch in range(epochs):
        benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                res = tm.fit(
                    train_x.astype(np.uint32),
                    train_y.astype(np.uint32),
                    metrics=["update_p"],
                )

            experiment_results["train_time"].append(benchmark1.elapsed())

            # print(res)
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(test_x) == test_y).mean()
                experiment_results["accuracy"].append(result)
            experiment_results["test_time"].append(benchmark2.elapsed())

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

    _LOGGER.info(f"Accuracies:{experiment_results['accuracy']}")
    return experiment_results

def main_coalesced(s, n_clauses, threshold, max_included_literals, weighted_clauses, platform, epochs, focused_negative_sampling):
    experiment_results = metrics_coalesced()
    
    '''train_x = np.load('tm_data/train_x.npy')
    train_y = np.load('tm_data/train_y.npy')
    test_x = np.load('tm_data/test_x.npy')
    test_y = np.load('tm_data/test_y.npy')'''
    
    tm = TMCoalescedClassifier(
        number_of_clauses=n_clauses,
        T=threshold,
        s=s,
        platform=platform,
        weighted_clauses=weighted_clauses,
        focused_negative_sampling=focused_negative_sampling
    )

    _LOGGER.info(f"Running {TMCoalescedClassifier} for {epochs}")
    for epoch in range(epochs):

        benchmark1 = BenchmarkTimer()
        with benchmark1:
            tm.fit(train_x, train_y)
        experiment_results["train_time"].append(benchmark1.elapsed())

        benchmark2 = BenchmarkTimer()
        with benchmark2:
            result = 100 * (tm.predict(test_x) == test_y).mean()
            experiment_results["accuracy"].append(result)
        experiment_results["test_time"].append(benchmark2.elapsed())

        number_of_positive_clauses = 0
        for i in range(tm.number_of_classes):
            number_of_positive_clauses += (tm.weight_banks[i].get_weights() > 0).sum()
        number_of_positive_clauses /= tm.number_of_classes
        experiment_results["number_of_positive_clauses"].append(number_of_positive_clauses)

        number_of_includes = 0
        for j in range(n_clauses):
            number_of_includes += tm.number_of_include_actions(j)
        number_of_includes /= 2 * n_clauses
        experiment_results["number_of_includes"].append(number_of_includes)

        _LOGGER.info(
            f"Epoch: {epoch + 1}, "
            f"Accuracy: {result:.2f}, "
            f"Positive clauses: {number_of_positive_clauses}, "
            f"Literals: {number_of_includes}, "
            f"Training Time: {benchmark1.elapsed():.2f}s, "
            f"Testing Time: {benchmark2.elapsed():.2f}s"
        )

    return experiment_results

def objective(trial):
    train_x = np.load('tm_data/train_x.npy')
    s = trial.suggest_float('s', 2.0, 20.0)
    n_clauses = trial.suggest_int('n_clauses', 100, 10000, step=2)
    threshold = trial.suggest_int('threshold', 3, 10000)
    max_included_literals = trial.suggest_int('max_included_literals', 1, train_x.shape[1])
    weighted_clauses = trial.suggest_categorical('weighted_clauses', ['True','False'])
    epochs = 12
    tm_type=args.tm_type
    #tm_type = trial.suggest_categorical('tm_type', ['vanilla','coalesced'])

    _LOGGER.info(f"Performing hp search with args: s={s}, T={threshold}, n_clauses={n_clauses}, max_included_literals={max_included_literals}, weighted_clauses={weighted_clauses}, tm_type={tm_type} ")
    try:
        if tm_type=='vanilla':
            platform = "GPU"
            exp_res = main(s, n_clauses, threshold, max_included_literals, weighted_clauses, platform, epochs)
        elif tm_type=='coalesced':
            platform = "CPU"
            focused_negative_sampling = trial.suggest_categorical('focused_negative_sampling', ['True','False'])
            exp_res = main_coalesced(s, n_clauses, threshold, max_included_literals, weighted_clauses, platform, epochs, focused_negative_sampling)
    
        return max(exp_res['accuracy'])
    except Exception as e:
        print(e)
        return 70.0
    
if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default='orig', type=str)
    parser.add_argument("--upscale", default=False, type=bool)
    parser.add_argument("--downscale", default=False, type=bool)
    parser.add_argument("--tm_type", default='vanilla', type=str)
    parser.add_argument("--num_classes", default=3, type=int)
    args = parser.parse_args()
    
    if args.data_type=='orig':
        train_x, train_y, test_x, test_y = load_orig_data(upscale=args.upscale,downscale=args.downscale,save=True,num_classes=args.num_classes)
    elif args.data_type=='processed':
        train_x, train_y, test_x, test_y = load_data(upscale=args.upscale,downscale=args.downscale,save=True,num_classes=args.num_classes)
    
    study = optuna.create_study(study_name=f"hp_{args.tm_type}_{args.upscale}_{args.downscale}_{args.data_type}.pkl", direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=1, show_progress_bar=True)
    print(study.best_params)
    print(study.best_value)
    pickle.dump(study.best_params, open('params.pkl','wb'))
    pickle.dump(study.best_value, open('performance.pkl','wb'))

    _LOGGER.info(study.best_params)
    _LOGGER.info(study.best_value)
    joblib.dump(study, f"hpsearch/hp_{args.tm_type}_{args.upscale}_{args.downscale}_{args.data_type}_{args.num_classes}_testbest.pkl")
    study = joblib.load(f"hpsearch/hp_{args.tm_type}_{args.upscale}_{args.downscale}_{args.data_type}_{args.num_classes}_testbest.pkl")
    
    save_path = f"plots/hp_{args.tm_type}_{args.upscale}_{args.downscale}_{args.data_type}_{args.num_classes}_testbest"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fig = optuna.visualization.plot_optimization_history(study)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"{save_path}/optimization_history.png")  
    
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(f"{save_path}/parallel_coordinate.png")
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f"{save_path}/param_importances.png")

