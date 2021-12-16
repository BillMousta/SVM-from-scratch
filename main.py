import SupportVectorMachine
import numpy as np
# from libsvm.svmutil import *
import DataProcessing
import time
from sklearn import svm
import Visualizations

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

    # param_grid = {'C': [0.1, 0.01, 0.001], 'kernel': ['linear', 'polynomial', 'RBF'], 'gamma': [2, 3, 4]}

    C = [16, 32, 64, 128, 256]
    # C = [0.001, 0.01, 0.1, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    # kernel = ['linear_kernel', 'polynomial_kernel', 'RBF']
    kernel = ['RBF']
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
        # clf = svm.SVC(kernel= 'rbf', C=0.1, gamma=0.1)
        # clf.fit(np.array(X_train), np.array(y_train))
        # predicted = clf.predict(X_test)
        # print(len(predicted))
        # print(predicted[predicted==-1].size)
        accuracy = accuracy_metric(np.array(y_test), predicted)
        scores.append(accuracy)

    print(scores)

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
"""
def compare_with_sklearn_svm():
    # Get train data 2000-2020 and testing data 2020-2021 for SP500
    processing = DataProcessing
    X_train, y_train = processing.processing_data('SPY', start='2000-01-01', end='2020-12-31', interval='1d')
    X_test, y_test = processing.processing_data('SPY', start='2021-01-01', end='2021-12-01', interval='1d')

    print(X_train.shape)
    Visualizations.visualize_class_distribution(y_train ,'test')

    print("----- My SVM -----")
    start = time.time()
    # My SVM with best parameters after grid search for this
    # specific data
    my_svm = SupportVectorMachine.SVM(kernel='RBF', C=128, gamma=0.5)
    my_svm.train(X_train, y_train)
    predicted = my_svm.predict(X_test)
    accuracy = accuracy_metric(y_test, predicted)
    print("Execution time for my SVM: {:.2f} sec".format(time.time() - start))

    print('Score: {:.2f}%'.format(accuracy))

    # SVM from sklearn library
    start = time.time()
    print("----- Sklearn SVM -----")
    sklearn_svm = svm.SVC(kernel= 'rbf', C=128, gamma=0.5)
    sklearn_svm.fit(np.array(X_train), np.array(y_train))
    predicted = sklearn_svm.predict(X_test)
    accuracy = accuracy_metric(y_test, predicted)
    print('Score: {:.2f}%'.format(accuracy))
    print("Execution time for sklearn SVM: {:.2f} sec".format(time.time() - start))



def run():
    # save_scores('test', 0.1, 1, 'RBF', 3)
    processing = DataProcessing
    n_folds = 5
    # data, classes = processing.processing_data('SPY', start='2000-01-01', end='2020-12-31', interval='1d')
    # grid_search(data, classes, n_folds)
    # scores = evaluate_algorithm(data, classes,  kernel='RBF', gamma=0.1, C=0.1, n_folds=5)
    # print(scores)
    compare_with_sklearn_svm()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()


