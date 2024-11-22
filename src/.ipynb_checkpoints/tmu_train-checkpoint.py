import os
import scipy
import pickle
import logging
import argparse
import warnings

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


_LOGGER = logging.getLogger(__name__)

def metrics(args):
    return dict(
        accuracy=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )

def metrics_coalesced():
    return dict(
        number_of_positive_clauses=[],
        accuracy=[],
        number_of_includes=[],
        train_time=[],
        test_time=[],
        args=vars(args)
    )

def main(args):
    experiment_results = metrics(args)
    
    '''train_x = np.load('tm_data/train_x.npy')
    train_y = np.load('tm_data/train_y.npy')
    test_x = np.load('tm_data/test_x.npy')
    test_y = np.load('tm_data/test_y.npy')'''

    if args.data_type == 'orig':
        train_x, train_y, test_x, test_y = load_orig_data(upscale=True, num_classes=3)
    elif args.data_type == 'processed':
        train_x, train_y, test_x, test_y = load_data(upscale=True)
    
    tm = TMClassifier(
        type_iii_feedback=False,
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    best_acc = 0
    for epoch in range(args.epochs):
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

            y_preds = tm.predict(test_x)
            accuracy = (y_preds == test_y).mean()
            print('accuracy:', accuracy)
            print('recall:', recall_score(test_y, y_preds, average='macro'))
            print('precision:', precision_score(test_y, y_preds, average='macro'))
            print('f1:', f1_score(test_y, y_preds, average='macro'))
            
            if accuracy>best_acc:
                best_acc=accuracy
                
                cm = confusion_matrix(test_y, y_preds)
                plt.figure(figsize=(12,10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix')
                plt.savefig(f'plots/conf_matrix/train_best.png')
            

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")
    print("BEST ACC:",best_acc)
    return experiment_results


def main_coalesced(args):
    experiment_results = metrics_coalesced()
    
    if args.data_type == 'orig':
        train_x, train_y, test_x, test_y = load_orig_data(upscale=True, num_classes=2)
    elif args.data_type == 'processed':
        train_x, train_y, test_x, test_y = load_data(upscale=True, num_classes=2)
    
    tm = TMCoalescedClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
        focused_negative_sampling=False
    )
    best_acc=0
    _LOGGER.info(f"Running {TMCoalescedClassifier} for {args.epochs}")
    for epoch in range(args.epochs):

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
        for j in range(args.num_clauses):
            number_of_includes += tm.number_of_include_actions(j)
        number_of_includes /= 2 * args.num_clauses
        experiment_results["number_of_includes"].append(number_of_includes)
        y_preds = tm.predict(test_x)
        accuracy = (y_preds == test_y).mean()
        print('accuracy:', accuracy)
        print('recall:', recall_score(test_y, y_preds, average='macro'))
        print('precision:', precision_score(test_y, y_preds, average='macro'))
        print('f1:', f1_score(test_y, y_preds, average='macro'))
        
        if accuracy>best_acc:
            best_acc=accuracy
            
            cm = confusion_matrix(test_y, y_preds)
            plt.figure(figsize=(10,10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(f'plots/conf_matrix/train_best.png')
        
        _LOGGER.info(
            f"Epoch: {epoch + 1}, "
            f"Accuracy: {result:.2f}, "
            f"Positive clauses: {number_of_positive_clauses}, "
            f"Literals: {number_of_includes}, "
            f"Training Time: {benchmark1.elapsed():.2f}s, "
            f"Testing Time: {benchmark2.elapsed():.2f}s"
        )
    print("BEST ACCURACY:", best_acc)
    return experiment_results

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=5000, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--platform", default="CPU", type=str, choices=["CPU", "CPU_sparse", "CUDA"])
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--tm_type", default="vanilla", type=str)
    parser.add_argument("--data_type", default="orig", type=str)
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args


if __name__ == "__main__":

    args = default_args()
    if args.tm_type == 'vanilla':
        results = main(args)
    elif args.tm_type == 'coalesced':
        results = main_coalesced(args)
    
    # python src/tmu_train.py --s 14.803539273459574 --n_clauses 8590 --threshold 3535 --max_included_literals 604 --weighted_clauses False
    
    # hp search with orig data
    #params = {'s': 14.803539273459574, 'n_clauses': 8590, 'threshold': 3535, 'max_included_literals': 604, 'weighted_clauses': 'False'}
    # (acc: 76.95):
    # {'s': 8.727282390265822, 'n_clauses': 5760, 'threshold': 5533, 'max_included_literals': 1142, 'weighted_clauses': 'False'}
    # python src/tmu_train.py --s 14.803539273459574 --n_clauses 8590 --threshold 3535 --max_included_literals 604 --weighted_clauses False

    # hp search with preprocessed data but kept columns
    #{'s': 5.570701569335727, 'n_clauses': 9904, 'threshold': 9692, 'max_included_literals': 1365, 'weighted_clauses': 'True'}

    # hp search TM Coalesced 
    # {'s': 14.348968566286867, 'n_clauses': 7408, 'threshold': 6542, 'max_included_literals': 552, 'weighted_clauses': 'True', 'focused_negative_sampling': 'False'}

    # hp search TM Coalesced with orig data (acc: 77.17)
    # {'s': 10.325687598172639, 'n_clauses': 6904, 'threshold': 4913, 'max_included_literals': 1562, 'weighted_clauses': 'True', 'focused_negative_sampling': 'False'}
    _LOGGER.info(results)


