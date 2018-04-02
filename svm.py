import numpy as np
import csv



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


def main():
    print(get_data("data/train.csv", 5))


if __name__ == "__main__":
    main()


