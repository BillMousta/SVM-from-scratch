import SupportVectorMachine
import numpy as np
# from libsvm.svmutil import *
import DataProcessing
import time
from sklearn import svm
import Visualizations
from sklearn.metrics import f1_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.pipeline import make_pipeline
import TestClassifiers
"""
Saving results of GridSearch into a txt file
"""
def save_scores(filename, score, kernel, C, gamma):
    file = open(filename, 'a')
    if kernel == 'RBF':
        L = ['C = '+str(C), ', kernel = ' + kernel, ', gamma ' + str(gamma), '\n']
        file.writelines(L)
        file.write('Score = ' + str(score) + '\n')
    else:
        L = ['C = ' + str(C), ', kernel = ' + kernel, '\n']
        file.writelines(L)
        file.write('Score = ' + str(score) + '\n')


"""
Algorithm for grid search
"""
def grid_search(dataset, classes, n_folds):
    fold_data, fold_classes = DataProcessing.cross_validation_split(dataset, classes,  n_folds)

    C = [0.001, 0.01, 0.1, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    kernel = ['linear_kernel', 'polynomial_kernel', 'RBF']
    gamma = [0.01, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    max_score = -100
    start = time.time()
    # for fixed 1 parameter test the other parameters
    for i in kernel:
        prev_score_C = 0.0
        for j in C:
            if i == 'RBF':
                prev_score_gamma = 0.0
                for g in gamma:
                    # returning the average of k scores for fixed parameters
                    fixed_scores_par = evaluate_algorithm(fold_data, fold_classes, kernel=i, C=j, gamma=g, n_folds=n_folds)
                    save_scores(i + ' Results', fixed_scores_par, i, j, g)

                    # replace with the best score
                    if fixed_scores_par > max_score:
                        max_score = fixed_scores_par
                        best_kernel = i
                        best_C = j
                        best_gamma = g
                    # hold the current score to compare it with the previous
                    # This is happen cause convex optimization has 1 solution
                    if fixed_scores_par < 1.001*prev_score_gamma:
                        break
                    else:
                        prev_score_gamma = fixed_scores_par
            else:
                # returning the average of k scores for fixed parameters
                fixed_scores_par = evaluate_algorithm(fold_data, fold_classes, kernel=i, C=j, gamma=0, n_folds=n_folds)
                save_scores(i + ' Results', fixed_scores_par, i, j, 0)
                # replace with the best score
                if fixed_scores_par > max_score:
                    max_score = fixed_scores_par
                    best_kernel = i
                    best_C = j
                    best_gamma = 0
                if fixed_scores_par < 1.01*prev_score_C:
                    break
                else:
                    prev_score_C = fixed_scores_par

    end = time.time()

    print(best_kernel)
    if best_kernel == 'RBF':
        print("Best parameters is: {} C = {} gamma = {}" .format(best_kernel,best_C, best_gamma))
        print('With score: {:.2f}%'.format(max_score))
    else:
        print("Best parameters is: {} C = {}".format(best_kernel, best_C))
        print('With score: {:.2f}%'.format(max_score))

    print("Time execution for grid search is: {}".format(end - start))

"""
Using k folds algorithm in SVM
"""
def evaluate_algorithm(fold_data, fold_classes, kernel, gamma, C, n_folds):
    # fold_data, fold_classes = DataProcessing.cross_validation_split(fold_data, fold_classes,  n_folds)
    scores = list()
    for i, fold in enumerate(fold_data):

        X_train = list(fold_data)
        X_train.pop(i)

        y_train = list(fold_classes)
        y_train.remove(fold_classes[i])

        # merge data into 1 list
        X_train = sum(X_train, [])
        y_train = sum(y_train, [])

        X_test = list()
        y_test = list(fold_classes[i])
        for row in fold:
            X_test.append(row)

        svm = SupportVectorMachine.SVM(kernel, C=C, gamma=gamma)
        svm.train(np.array(X_train), np.array(y_train))
        predicted = svm.predict(np.array(X_test))
        accuracy = accuracy_metric(np.array(y_test), predicted)
        scores.append(accuracy)

    # average score for k folds
    return sum(scores)/n_folds

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

"""
Compare results my svm class with sklearn svm 
with train data from sp500 2000-2021  and testing data 2021-now
"""
def compare_with_sklearn_svm(X_train, y_train, X_test, y_test):
    print("----- My SVM -----")
    start = time.time()
    # My SVM with best parameters after grid search for this
    # specific data
    my_svm = SupportVectorMachine.SVM(kernel='RBF', C=128, gamma=0.5)
    my_svm.train(X_train, y_train)
    predicted = my_svm.predict(X_test)
    accuracy = accuracy_metric(y_test, predicted)
    f1 = f1_score(y_test, predicted)
    print("Execution time for my SVM: {:.2f} sec".format(time.time() - start))

    print('Accuracy: {:.2f}%'.format(accuracy))
    print("F1 score: {:.2f}%".format(f1*100))

    # SVM from sklearn library
    start = time.time()
    print("----- Sklearn SVM -----")
    sklearn_svm = svm.SVC(kernel= 'rbf', C=128, gamma=0.5)
    sklearn_svm.fit(np.array(X_train), np.array(y_train))
    predicted = sklearn_svm.predict(X_test)
    accuracy = accuracy_metric(y_test, predicted)
    f1 = f1_score(y_test, predicted)
    print("Execution time for sklearn SVM: {:.2f} sec".format(time.time() - start))

    print('Score: {:.2f}%'.format(accuracy))
    print("F1 score: {:.2f}%".format(f1*100))


def precision_recall(X_train, y_train, X_test, y_test):
    sklearn_svm = svm.SVC(kernel='rbf', C=128, gamma=0.5)
    sklearn_svm.fit(np.array(X_train), np.array(y_train))
    pred = sklearn_svm.predict(X_test)
    display = PrecisionRecallDisplay.from_predictions(y_test, pred,name='rbf SVC')
    _ = display.ax_.set_title("2-class Precision-Recall curve")


def run():
    processing = DataProcessing
    # Get train data 2000-2020 and testing data 2021-now for SP500
    X_train, y_train = processing.processing_data('SPY', start='2000-01-01', end='2020-12-31', interval='1d')
    X_test, y_test = processing.processing_data('SPY test', start='2021-01-01', end='2021-12-01', interval='1d')
    n_folds = 5
    # Grid search for best parameters
    # grid_search(X_train, y_train, n_folds)

    compare_with_sklearn_svm(X_train, y_train, X_test, y_test)
    # precision_recall(X_train, y_train, X_test, y_test)

    # test other classifiers
    # TestClassifiers.testing_classifiers(X_train, y_train, X_test, y_test)

    print("Done!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()


