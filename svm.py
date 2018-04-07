import csv
import time
import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


def get_data(filename, reduction=None):
    with open(filename, 'r') as csvdata:
        rows_raw = csv.reader(csvdata)
        rows_np = np.array(list(rows_raw), dtype=int)

        if reduction is None:
            reduction = len(rows_np)

        print("Length of the data set: " + str(reduction))

        return {
            "X": rows_np[:reduction, 1:],
            "y": rows_np[:reduction, 0]
        }


def get_accuracy(predictions, solutions):
    return str((np.sum(predictions == solutions) / float(len(solutions))) * 100.0).join(' %')


def load_data(train_reduction, test_reduction):
    start_data = time.time()
    train_name = "data/train_serialized_" + str(train_reduction)
    test_name = "data/test_serialized_" + str(test_reduction)
    if Path(train_name).is_file() and Path(test_name).is_file():
        train_set = pickle.load(open(train_name, "rb"))
        test_set = pickle.load(open(test_name, "rb"))
        print("Loaded serialized files.")
    else:
        train_set = get_data("data/train.csv", train_reduction)
        pickle.dump(train_set, open(train_name, "wb"))
        test_set = get_data("data/test.csv", test_reduction)
        pickle.dump(test_set, open(test_name, "wb"))
        print("Parsed files and saved serialized files")
    end_data = time.time()
    print("Time to load data: %0.2fs" % (end_data - start_data) + "\n")
    return train_set, test_set


def train(model, train):
    start_training = time.time()
    model.fit(**train)
    model.score(**train)
    end_training = time.time()
    print("Time needed for training the model: %0.2fs" % (end_training - start_training))
    return model


def cross_validate(model, train):
    start_cv = time.time()
    scores = cross_val_score(model, **train, cv=5)
    end_cv = time.time()
    print("Time needed for cross validation: %0.2fs" % (end_cv - start_cv))
    print("Accuracy: %0.2f%% (+/- %0.2f%%)" % (scores.mean() * 100, scores.std() * 200))


def test(model, test):
    start_prediction = time.time()
    prediction = model.predict(test['X'])
    end_prediction = time.time()
    print("Time needed for prediction: %0.2fs" % (end_prediction - start_prediction))
    print("Accuracy: " + str(get_accuracy(prediction, test['y'])) + "\n")


def grid_search(model, train_set):
    '''
        Results:
        RBF: The best parameters are {'C': 10.0, 'gamma': 1e-07} with a score of 0.91
    '''
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv)
    grid.fit(**train_set)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))


def main(bagging):
    # here the size of the datasets can be determined. Influences the time of the algorithm
    train_set, test_set = load_data(10000, 7000)

    # Can be commented in to try different C values for linear kernel
    #C_values = np.logspace(-5, 10)
    #model = SVC(kernel='linear', cache_size=4000)
    #perform_c_parameter_search(C_values, model, test_set, train_set)

    model = SVC(kernel='rbf', cache_size=4000, gamma=1e-07, C=10)

    if bagging:
        estimators = 10
        model = BaggingClassifier(model, max_samples=1.0 / estimators, n_estimators=estimators, n_jobs=-1)
    perform_analysis(model, test_set, train_set)


def perform_c_parameter_search(C_values, model, test_set, train_set):
    for C in C_values:
        model.C = C  # decrease C if lots of noisy datapoints
        print("C = " + str(C))
        perform_analysis(model, test_set, train_set)


def perform_analysis(model, test_set, train_set):
    model = train(model, train_set)
    cross_validate(model, train_set)
    test(model, test_set)


if __name__ == "__main__":
    main(False)
