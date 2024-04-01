import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import timeit

from functools import partial

from decision_tree import *
from naive_bayes import *

def test_dt():
    with open('./datasets/test_dt.pkl', 'rb') as f:
        test = pickle.load(f)

    seed = 0
    num_classes = 2
    max_depth = 10
    min_leaf_data = 10
    min_entropy = 1e-3
    num_split_retries = 10
    debug = False

    # 1-D Linear Seperable
    X_1 = test['X_1']
    y_1 = test['y_1']
    tree = train_dt(X_1,
                    y_1,
                    seed,
                    max_depth,
                    min_leaf_data,
                    min_entropy,
                    num_split_retries,
                    num_classes,
                    debug)
    print('Correct Pred 1-D Linear Seperable, 2 class: {}'.format(np.allclose(predict_dt(tree, X_1), test['pred_1'])))
    # 1-D Random, 2 Class
    X_2 = test['X_2']
    y_2 = test['y_2']
    tree = train_dt(X_2,
                    y_2,
                    seed,
                    max_depth,
                    min_leaf_data,
                    min_entropy,
                    num_split_retries,
                    num_classes,
                    debug)
    print('Correct Pred 1-D Random, 2 class: {}'.format(np.allclose(predict_dt(tree, X_2), test['pred_2'])))

    # Optional TODO: Add your own test cases
def test_nb():
    with open('./datasets/test_nb.pkl', 'rb') as f:
        gt_data = pickle.load(f)
        data = gt_data["nb"]

    num_features = gt_data["X"].shape[1]
    num_classes = len(np.unique(gt_data["y"]))
    params = train_nb(gt_data["X"], gt_data["y"], num_classes)

    correct_means = np.allclose(params.means, data["means"])
    correct_covariances = np.allclose(params.covariances, data["covariances"])
    correct_priors = np.allclose(params.priors, data["priors"])

    correct_params = Params(data["means"],
                            data["covariances"],
                            data["priors"],
                            num_features,
                            num_classes)
    model_probs = predict_nb(correct_params, gt_data["X"])
    correct_predictions = np.allclose(model_probs, data["predictions"])

    print(f"Correct Means: {correct_means}")
    print(f"Correct Covariances: {correct_covariances}")
    print(f"Correct Priors: {correct_priors}")
    print(f"Correct Predictions: {correct_predictions}")

    print("Details:")
    if not correct_means:
        print("Expected Mean:")
        print(data["means"])
        print("Got:")
        print(params.means)

    if not correct_covariances:
        print("Expected Covariances:")
        print(data["covariances"])
        print("Got:")
        print(params.covariances)

    if not correct_priors:
        print("Expected Priors:")
        print(data["priors"])
        print("Got:")
        print(params.priors)

    if not correct_predictions:
        print("Expected Predictions:")
        print(data["predictions"])
        print("Got:")
        print(model_probs)

    print("=" * 75)

    # Optional TODO: Add your own test cases
test_dt()
test_nb()
