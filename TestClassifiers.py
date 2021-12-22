import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
import Visualizations
import time
import Data

class MultiLayerPerceptron(nn.Module):

    def __init__(self, size_features, size_nodes):
        super(MultiLayerPerceptron, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(size_features, size_nodes),
            nn.Dropout(0.25),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(size_nodes, size_nodes),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Dropout(0.25),
            # nn.Linear(size_nodes, size_nodes),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(size_nodes, 2),
        )

    def forward(self, x):
        logits = self.layer(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    correct = 0
    pred_training_labels = []
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        for label in pred.argmax(1):
            num = label.numpy()
            pred_training_labels .append(num)

        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        # sets the gradients to zero before we start backpropagation.
        # This is a necessary step as PyTorch accumulates the gradients from
        # the backward passes from the previous epochs.

        optimizer.zero_grad()
        # computes the gradients
        loss.backward()
        # updates the weights accordingly
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss /= num_batches

    return pred_training_labels, correct/size, train_loss


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    labels = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            for label in pred.argmax(1):
                labels.append(label.numpy())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct, labels


"""
K nearest neighbor algorithm using k = 3, to find accuracy
for stock data test
"""
def k_nearest_neighbors(X_train, y_train, X_test, y_test):
    start_time = time.time()
    print("----- 3 Nearest Neighbors -----")
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    train_preds = knn_model.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print("For training data")
    print("The square root error is: {:.4f}".format(rmse))
    print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_train,train_preds)*100))
    test_preds = knn_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)

    print("For testing data")
    print("The square root error for testing data is: {:.4f}".format(rmse))
    print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_test, test_preds) * 100))

    print("--- %s seconds ---" % (time.time() - start_time))

"""
Nearest neighbor algorithm, to find accuracy
for stock data test
"""
def nearest_centroid(X_train, y_train, X_test, y_test):
    start_time = time.time()
    print("----- Nearest Centroids -----")
    centroid_model = NearestCentroid(metric='euclidean', shrink_threshold=None)
    centroid_model.fit(X_train, y_train)
    train_preds = centroid_model.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print("For training data")
    print("The square root error is: {:.4f}".format(rmse))
    print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_train, train_preds) * 100))

    test_preds = centroid_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print("For testing data")
    print("The square root error for testing data is: {:.4f}".format(rmse))
    print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_test, test_preds) * 100))
    print("--- %s seconds ---" % (time.time() - start_time))


"""
Run multi layer perceptron
"""
def mlp(X_train, y_train, X_test, y_test, num_features):
    train_data = Data.Data(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_data = Data.Data(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    batch_size = 32
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    visual = Visualizations
    torch.manual_seed(213)
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    start_time = time.time()
    model = MultiLayerPerceptron(num_features, 256).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)

    # testing_learning_rates(train_dataloader,test_dataloader,device,visual)

    epochs = 100
    # the below lists help us to visualize the results
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    labels_train = []
    labels_test = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        labels_train, acc, train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(acc)
        test_loss, accuracy, labels_test = test(test_dataloader, model, loss_fn, device)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

    print("--- %s seconds ---" % (time.time() - start_time))

def testing_classifiers(X_train, y_train, X_test, y_test):
    mlp(X_train, y_train, X_test, y_test, 7)
    k_nearest_neighbors(X_train, y_train, X_test, y_test)
    nearest_centroid(X_train, y_train, X_test, y_test)
