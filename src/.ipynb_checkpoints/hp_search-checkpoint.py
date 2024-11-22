from green_tsetlin.hpsearch import HyperparameterSearch
import numpy as np
import pickle

train_x = np.load('tm_data/train_x.npy')
train_y = np.load('tm_data/train_y.npy')
test_x = np.load('tm_data/test_x.npy')
test_y = np.load('tm_data/test_y.npy')

hyperparam_search = HyperparameterSearch(s_space=(2.0, 20.0),
                                        clause_space=(5, 10000),
                                        threshold_space=(3, 10000),
                                        max_epoch_per_trial=10,
                                        literal_budget=(1, train_x.shape[1]),
                                        seed=42,
                                        n_jobs=128,
                                        k_folds=2,
                                        minimize_literal_budget=False)

hyperparam_search.set_train_data(train_x, train_y)
hyperparam_search.set_eval_data(test_x, test_y)

hyperparam_search.optimize(n_trials=10, study_name='ikt463_search', show_progress_bar=True, storage=None)

params = hyperparam_search.best_trials[0].params
performance = hyperparam_search.best_trials[0].values

print('best params:', params)
print('best score:', performance)

pickle.dump(params, open('params.pkl','wb'))
pickle.dump(performance,open('performance.pkl','wb'))
