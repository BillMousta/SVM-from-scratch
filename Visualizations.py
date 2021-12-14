import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_class_distribution(data,title):
    sns.set_theme()
    dict = {'Buy': data[data==1].size, 'Sell': data[data==-1].size}
    fig = plt.figure(figsize=(12, 8))

    plt.title(title)
    plt.bar('Sell', height=dict['Sell'], color='r', label='Sell')
    plt.bar('Buy', height=dict['Buy'], color='b', label='Buy')
    plt.xlabel('Movement')
    plt.ylabel('No of Labels')
    plt.legend()
    # plt.savefig(title)
    plt.show()

def histogram(data):
    sns.set_theme()
    fig = plt.figure(figsize=(12,10), dpi=200)
    plt.hist(data['Daily Close Return'], bins=100)
    plt.title('Distribution of Daily Percent Change of Close Price')
    plt.show()

def visualize_data(data, y, title):
    fig = plt.figure(figsize=(12,8), dpi=200)
    X, Y = np.split(data,[-1],axis=1)
    plt.plot(X, Y, 'o', label='Class 1', markevery=(y == 1))
    plt.plot(X, Y, 'o', label='Class -1', markevery=(y == -1))
    plt.title(title + " Data after standardization")
    plt.xlabel('Price')
    plt.ylabel('Moving Average 7')
    plt.legend()
    plt.show()
    # plt.plot(data['Daily Return'], 'o', c='r',label='Class 1', markevery=(data['Label']==1))
    # plt.plot(data['Daily Return'], 'o', c='b', label='Class -1', markevery=(data['Label'] == -1))
    # plt.legend()
    # plt.show()