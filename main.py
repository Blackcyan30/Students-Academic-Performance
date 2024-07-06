import os, random
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import ModelCheckpoint
from keras.models import  load_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score

seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
# For working on GPUs from "TensorFlow Determinism"
os.environ["TF_DETERMINISTIC_OPS"] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

categorical_attr = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class']

df = pd.read_csv('Dataset.csv')


le = LabelEncoder()
df[categorical_attr] = df[categorical_attr].apply(le.fit_transform, axis=0)

# X: Features, y: Classes
X = np.array(df.iloc[:, :-1])
y = np.array(df['Class'])

X_orig_without_normalization = X

# Normalization
X = tf.keras.utils.normalize(X, axis=-1, order=2)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12, stratify=y)

# Initialize SVM classifier
svm = SVC(kernel='linear', C=1.0)

# Train the classifier
svm.fit(X_train, y_train)

# Predict the labels of test data
y_pred = svm.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)

print("Accuracy:", accuracy)

def learn_classifier(X_train, y_train, kernel):
    """
    This function learns the classifier from the input features and labels using the 
    kernel function supplied
    Inputs:
        X_train
        y_train
        kernel
    Outputs:
        classifier learned from data
    """

    clf = SVC(kernel=kernel)

    clf.fit(X_train, y_train)

    return clf

clf = learn_classifier(X_train, y_train, 'rbf')
y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy using learn_classifier function: {accuracy}")

def evaluate_classifier(clf, X_validation, y_validation):
    """
    Evaluates the classifier based on the supplied validation data.
    Inputs:
        clf: classifier to evaluate
        X_validation: feature matrix
        y_validation: class labels
    Outputs:
        double: accuracy of classifier on validation data
    """
    y_pred = clf.predict(X_validation)
    
    accuracy = accuracy_score(y_validation, y_pred)

    return accuracy

accuracy = evaluate_classifier(clf, X_train, y_train)
print(f"Accuracy using the evaluate_classifier function: {accuracy}")

## **Doing KFold cross-validation**

kf = KFold(n_splits=5, random_state=1, shuffle=True)
print(f"KFold cross-validation object being used: {kf}")

def best_model_selection(kf, X, y):
    """
    This function selects the kernel giving the best results using 
    k-fold cross-validation
    Input:
        kf: object defined above
        X: training data
        y: training labels
    Return:
        best_kernel (string)
    """

    kernel_accuracy = {}

    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
        accuracies = []
        i = 1
        for train_index, test_index in kf.split(X):
            
            # Splitting data into training and testing sets for the 
            # current fold.
            X_train, y_train= X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            classifier = learn_classifier(X_train, y_train, kernel)
            accuracy = evaluate_classifier(classifier, X_test, y_test)

            # Appending the accuracy/error to the error array for 
            # further evaluation of mean error.
            print(f"The kernel {kernel} has the accuracy of {accuracy} in the fold{i}")
            i += 1
            accuracies.append(accuracy)

        # Calculating teh average accuracy for the current kernel 
        # across all folds
        kernel_accuracy[kernel] = np.mean(accuracies)
        print(f"\nThe kernel '{kernel}' has the accuracy of: {np.mean(accuracies)}\n")
    
    # Getting the best kernel with the highest average accuracy.
    best_kernel = max(kernel_accuracy, key=kernel_accuracy.get)
    print(f"Selected best kernel is: {best_kernel}")

    # Returning the best kernel as a string.
    return best_kernel


best_kernel = best_model_selection(kf, X_train, y_train)
print(f"The best kernel found after KFold is: {best_kernel}")

