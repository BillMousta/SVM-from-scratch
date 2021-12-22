from random import randrange
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import Visualizations
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns

def store_data(name, start, end, interval):
    filename = name + '.csv'
    # filename = 'SPY test.csv'
    # data = yf.download(name, start=start, end=end, interval=interval)
    # Write data
    # data.to_csv(filename, encoding='utf-8', date_format='%Y-%m-%d')
    # Read from .csv file instead of yahoo finance
    data = pd.read_csv(filename, parse_dates=True, index_col='Date')
    # print(data[['Adj Close', 'Close']])
    data = add_features(data)
    # Adding Labels
    data = add_labels(data)
    # data = data.drop(['Adj Close'], axis=1)
    data.drop(['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
    return data

"""
Adding extra features
"""
def add_features(data):
    data['Daily Close Return'] = data['Close'].pct_change(1)
    data['Daily High Return'] = data['High'].pct_change(1)
    data['Daily Low Return'] = data['Low'].pct_change(1)
    data['Daily High-Close Return'] = (data['High'] - data['Close'])/data['Close']
    data['Daily Close-Low Return'] = (data['Close'] - data['Low'])/data['Close']
    data['Daily Volume'] = data['Volume'].pct_change(1)
    data['MA 7'] = data['Close'].rolling(7).mean().shift(-7)
    # Visualizations.histogram(data['Close'], 'Distribution of Closing Price')
    # Visualizations.histogram(data['Daily Close Return'], 'Distribution of Daily Return of Closing Price')
    data.dropna(inplace=True)
    data.drop(data.index[0], inplace=True)
    return data

"""
Turn finance data into binary classification problem
according with: if MA 7.shift(-7) > 0.003
                    1 
                else 
                    -1 
"""
def add_labels(data):
    # print(data['Adj Close'])
    movements = []
    movement = (data['MA 7'] - data['Close'])/ (data['Close'])
    # print(movement)
    for move in movement:
        # Buying signal
        if move > 0.002:
            movements.append(1)
        else:  # Selling signal
            movements.append(-1)

    data['Label'] = movements
    # data.drop(['Adj Close', 'MA 7'], axis=1, inplace=True)
    return data


def processing_data(name, start, end, interval):
    data = store_data(name, start=start, end=end, interval=interval)

    # Visualizations.visualize_class_distribution(data['Label'], 'Test Distribution')

    y = data['Label'].to_numpy()
    data.drop(['Label'], axis=1, inplace=True)
    X = data.to_numpy()

    # Standardlization
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    # Visualizations.visualize_data(np.vstack((X[0:, 0], X[0:, 5])), y, 'Train data visualization')
    # print(X)
    # print(X[0:,0])
    # print(X[0:,5])

    return X, y


"""
# Split a dataset into k folds in a list
"""
def cross_validation_split(dataset, labels, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    labels_split = list()
    labels_copy = list(labels)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold_data = list()
        fold_label = list()
        while len(fold_data) < fold_size:
            index = randrange(len(dataset_copy))
            fold_data.append(dataset_copy.pop(index))
            fold_label.append(labels_copy.pop(index))

        dataset_split.append(fold_data)
        labels_split.append(fold_label)

    return dataset_split, labels_split


if __name__ == '__main__':
    visual = Visualizations
    processing_data('SPY', start='2000-01-01', end='2021-12-01', interval='1d')
    sp500 = store_data('SPY', start='2000-01-01', end='2021-01-01', interval='1d')
    # print(sp500)
    # visual.visualize_data(sp500)
    # visual.visualize_class_distribution(sp500['Label'], 'Classes')
    # visual.histogram(sp500)
