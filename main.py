import SupportVectorMachine
import numpy as np
# from libsvm.svmutil import *
import DataProcessing
import time

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

    C = [0.001, 0.01, 0.1, 1, 2, 4, 8, 16, 32, 64, 128]
    kernel = ['linear_kernel', 'RBF']
    gamma = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    max_score = -100
    start = time.time()
    # for fixed 1 parameter test the other parameters
    for i in kernel:
        for j in C:
            if i == 'RBF':
                for g in gamma:
                    # returning the average of k scores for fixed parameters
                    fixed_scores_par = evaluate_algorithm(fold_data, fold_classes, i, j, g, n_folds)
                    save_scores('RBF Results', fixed_scores_par, i, j, g)
                    # replace with the best score
                    if fixed_scores_par > max_score:
                        max_score = fixed_scores_par
                        best_kernel = i
                        best_C = j
                        best_gamma = g
            else:
                # returning the average of k scores for fixed parameters
                fixed_scores_par = evaluate_algorithm(fold_data, fold_classes, i, j, 0, n_folds)
                save_scores('Linear Results', fixed_scores_par, i, j, 0)
                # replace with the best score
                if fixed_scores_par > max_score:
                    max_score = fixed_scores_par
                    best_kernel = i
                    best_C = j
                    best_gamma = 0
    end = time.time()

    print(best_kernel)
    if best_kernel == 'RBF':
        print("Best parameters is: {} C = {} gamma = {}" .format(best_kernel,best_C, best_gamma))
        print('With score: {:.2f}%'.format(max_score*100))
    else:
        print("Best parameters is: {} C = {}".format(best_kernel, best_C))
        print('With score: {:.2f}%'.format(max_score * 100))

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
        print(len(predicted))
        print((predicted==-1).size)
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


def run():
    # save_scores('test', 0.1, 1, 'RBF', 3)
    processing = DataProcessing
    n_folds = 5
    data, classes = processing.processing_data('SPY')
    grid_search(data,classes,n_folds)
    # scores = evaluate_algorithm(data, classes,  kernel='RBF', gamma=0, C=0.1, n_folds=5)
    # print(scores)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()


