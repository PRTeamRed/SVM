import numpy as np
import csv

''' inspired by:
    https://maviccprp.github.io/a-support-vector-machine-in-just-a-few-lines-of-python-code/
'''


def get_data(filename, red=None):
    with open(filename, 'r') as csvdata:
        rows_raw = csv.reader(csvdata)
        rows_np = np.array(list(rows_raw), dtype=int)
        
        if red is None:
            red = len(rows_np)
        print("length of the data set: " + str(red))

        #creating bias array
        samplesWithBias = np.negative(np.ones((red, len(rows_np[0])), dtype=int))

        #samples with bias
        samplesWithBias[:,:-1] = rows_np[:red, 1:]

        return {
                "samples": samplesWithBias,
                "labels": rows_np[:, 0]
                }


def svm(samples, labels, epochs=100):
    w = np.zeros(len(samples[0]))
    eta = 1
    errors = []

    for epoch in range(1, epochs):
        error = 0
        #TODO randomly pick a sample
        for i, x in enumerate(samples):
            if (labels[i] * np.dot(samples[i], w)) < 1:
                w = w + eta * ((samples[i] * labels[i]) + (-2 * (1/epoch) * w))
                error = 1
            else:
                w = w + eta * (-2 * (1/epoch) * w)
        errors.append({"error":error, "epoch": epoch})
    return {"w": w, "errors": errors}


def main():
    data = get_data("data/train.csv", 500)
    result = svm(**data, epochs=10000)
    print(len(result['errors']))

if __name__ == "__main__":
    main()


