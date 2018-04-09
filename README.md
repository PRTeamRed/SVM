# SVM
Implementing SVM for the course Pattern Recognition. The model is trained on the MI

### Requirements
Python 3.4 or higher is required and PIP.
You need to have a folder called "data" in the root folder that contains the train and test csv files.

To install all the needed dependencies execute this command in the root directory of the project:
```
pip install -r requirments.txt
```

### Reduce train and test set
The test and train set can be reduced by passing two parameters to the load_data() function. The first one reduces the train set to the given number and the second one the test set. If the two parameters are None it takes the whole set. 
To adapt the two sets just change the two parameters in line 95 of the svm.py file.

### Run
There are two implementations of the SVM: svm.py, which uses libs that provide an implementation of SVM and svm_v_01.py, which is our own implementation of SVM that just uses libs for the calculations.

**svm_v_01.py** is still under construction and is **NOT** working correctly.

To run them either call: 
```
python svm.py
```
or
```
python svm_v_01.py
```
