import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""
Visualizing the number of each class in binary
classification problem
"""
def visualize_class_distribution(data,title):
    sns.set_theme()
    dict = {'Buy': data[data==1].size, 'Sell': data[data==-1].size}
    fig = plt.figure(figsize=(12, 8))

    plt.title(title)
    plt.bar('Sell', height=dict['Sell'], color='r', label='Sell')
    plt.bar('Buy', height=dict['Buy'], color='b', label='Buy')
    plt.xlabel('Movement')
    plt.ylabel('No of Labels')
    plt.legend(loc=(1, 0.92))
    plt.show()

def histogram(data, title):
    sns.set_theme()
    # fig = plt.figure(figsize=(12,10), dpi=200)
    plt.hist(data, bins=50)
    plt.title(title)
    plt.show()

def visualize_data(data, y, title):
    sns.set_theme()
    fig = plt.figure(figsize=(12,8), dpi=200)
    # X, Y = np.split(data,[-1],axis=1)
    X = data[0, 0:]
    Y = data[1, 0:]
    plt.plot(X, Y, 'o', label='Class 1', markevery=(y == 1))
    plt.plot(X, Y, 'o', label='Class -1', markevery=(y == -1))
    plt.title(title + " Data after standardization")
    plt.xlabel('Daily Close Price')
    plt.ylabel('Moving Average 7')
    plt.legend()
    plt.show()