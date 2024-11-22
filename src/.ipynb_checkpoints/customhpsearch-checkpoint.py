import os
import pickle
import optuna

import numpy as np
import green_tsetlin as gt

from sklearn.metrics import recall_score, precision_score, f1_score

train_x = np.load('tm_data/train_x.npy')
train_y = np.load('tm_data/train_y.npy')
test_x = np.load('tm_data/test_x.npy')
test_y = np.load('tm_data/test_y.npy')

tm_data_dir = "tm_data"
tm_save_path = os.path.join(tm_data_dir, "tm_state_v1.npz")


def objective(trial):
    s = trial.suggest_float('s', 2.0, 20.0)
    n_clauses = trial.suggest_int('n_clauses', 5, 10000)
    threshold = trial.suggest_int('threshold', 3, 10000)
    literal_budget = trial.suggest_int('literal_budget', 1, train_x.shape[1])


    tm = gt.TsetlinMachine(n_literals=train_x.shape[1],
                           n_clauses=n_clauses,
                           n_classes=2,
                           s=s,
                           threshold=threshold,
                           literal_budget=literal_budget
                           )

    trainer = gt.Trainer(tm, seed=42, n_jobs=128, progress_bar=True, n_epochs=10)

    trainer.set_train_data(train_x, train_y)
    trainer.set_eval_data(test_x, test_y)

    trainer.train()

    tm.save_state(tm_save_path)

    tm.load_state(tm_save_path)
    predictor = tm.get_predictor()

    y_preds = []
    for x in test_x:
        y_pred = predictor.predict(x)
        y_preds.append(y_pred)
    
    acc = np.mean(y_preds == test_y)
    recall = recall_score(test_y, y_preds, average='macro')
    precision = precision_score(test_y, y_preds, average='macro')
    f1 = f1_score(test_y, y_preds, average='macro')

    return acc


if __name__=='__main__':
    study = optuna.create_study(study_name='ikt463_search2', storage=None, direction='maximize')
    study.optimize(objective, n_trials=10, n_jobs=1, show_progress_bar=True)
    print(study.best_params)
    print(study.best_value)
    pickle.dump(study.best_params, open('params.pkl','wb'))
    pickle.dump(study.best_value, open('performance.pkl','wb'))