import csv
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

def get_data(filename, red=None):
    with open(filename, 'r') as csvdata:
        rows_raw = csv.reader(csvdata)
        rows_np = np.array(list(rows_raw), dtype=int)[:1000]

        if red is None:
            red = len(rows_np)

        print("length of the data set: " + str(red))

        return {
                "X": rows_np[:red, 1:],
                "y": rows_np[:red, 0]
                }

def get_accuracy(predictions, solutions):
    return str((np.sum(predictions == solutions) / float(len(solutions))) * 100.0).join(' %')


def main():
    start_data = time.time()
    training_data = get_data("data/train.csv")
    test_data = get_data("data/test.csv")
    end_data = time.time()

    print("Time to load data: " + str(end_data - start_data))

    start_training = time.time()
    model = SVC(kernel='linear')
    #model = SVC(kernel='rbf', gamma=5)
    #model = SVC(kernel='sigmoid', coef0=0.5)
    model.cache_size = 2000
    model.C = 1 # decrease C if lots of noisy datapoints


    estimators = 10
    #model = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', C=50, gamma=1), max_samples = 1.0/estimators, n_estimators=estimators, n_jobs=-1))
    #model = RandomForestClassifier(min_samples_leaf=20)

    model.fit(**training_data)
    model.score(**training_data)

    end_training = time.time()

    print("Time to train the model: " + str(end_training - start_training))

    start_prediction = time.time()
    prediction = model.predict(test_data['X'])
    end_prediction = time.time()

    print("Time needed for prediction: " + str(end_prediction - start_prediction))
    print("Accuracy: " + str(get_accuracy(prediction, test_data['y'])))


if __name__ == "__main__":
    main()

