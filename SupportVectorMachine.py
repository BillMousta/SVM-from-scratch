import sys
sys.path.append('C:\\Users\\Bill Moustakidis\\anaconda3\\envs\\svm\\lib\\site-packages')
import cvxopt
import cvxopt.solvers
import numpy as np
import DataProcessing

class SVM(object):
    """
    Kernel parameter can be polynomial, RBF etc...
    The C parameter tells the SVM optimization
    how much you want to avoid misclassifying each training example
    tol is the tolerance for support vectors
    """
    def __init__(self, kernel, C, gamma, tol=1e-5):

        if kernel == 'RBF':
            self.kernel = RBF
            self.gamma = gamma
        elif kernel == 'linear_kernel':
            self.kernel = linear_kernel
        else:
            self.kernel = polynomial_kernel

        self.C = C
        self.tol = tol

    """
     Training model for SVM with cvxopt optimization   
    """
    def train(self,data, classes):
        samples, features = data.shape

        # Kernel or Gram matrix
        K = np.zeros((samples, samples))
        for i in range(samples):
            for j in range(samples):
                # Find similarities between samples in bigger dimensions
                # if data is  non-linear separable
                # K(xi,xj) = φ(xi)*φ(xj)^T where φ(.) is the kernel choice
                if self.kernel == RBF:
                    K[i][j] = self.kernel(data[i], data[j], self.gamma)
                else:
                    K[i][j] = self.kernel(data[i], data[j])

        """
        Duality Problem: 
        minimize 1/2 * x^T P x + q^T x 
        subject to Ax = b
                   0 <= x_i <= C 
         where Pij = yi * yj * K(xi, xj)
                y classes -1 or 1
                q = [1,1,...,1]^T
                b = [0,0,...,0]^T               
        """
        P = cvxopt.matrix(np.outer(classes, classes) * K)
        q = cvxopt.matrix(np.ones(samples)* -1)
        A = cvxopt.matrix(classes, (1,samples), 'd')
        b = cvxopt.matrix(0.0)

        """
        define Matrix G and h where
        Gx <= h
        where G is a matrix with values of variables of constraints 
        and h is a vector of constant values of constraints
        """
        con1 = np.identity(samples)*(-1)
        con2 = np.identity(samples)*(-1)
        G = cvxopt.matrix(np.vstack((con1,con2)))
        con1 = np.zeros(samples)
        con2 = np.ones(samples) * self.C
        h = cvxopt.matrix(np.hstack((con1,con2)))

        # Get the solution of the convex problem
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Get Lagrange multipliers
        a = np.ravel(sol['x'])

        # define the support vectors which have non zero
        # Lagrange multipliers
        support_vector = a > self.tol
        ind = np.arange(len(a))[support_vector]
        self.a = a[support_vector]
        self.support_vector = data[support_vector]
        self.support_vector_y = classes[support_vector]
        # print("%d support vectors out of %d points" % (len(self.a), samples))

        """
        Get the optimized w and b according with support vectors:
        w_opt = SUM(ai*yi*xi) for linear kernel
        b_opt = 1 - SUM(ai * yi * K(xi,xj)
        """
        if self.kernel == linear_kernel:
            self.w = np.zeros(features)
            for i in range(len(self.a)):
                self.w += self.a[i] * self.support_vector_y[i] * self.support_vector[i]
        else:
            self.w = None


        self.b = 0
        for n in range(len(self.a)):
            self.b += self.support_vector_y[n]
            self.b -= np.sum(self.a * self.support_vector_y * K[ind[n], support_vector])
        self.b /= len(self.a)

    """
       Decision rule if d_i(w^T * x_i + b) >= 1 class 1
                     else                       class -1
       where d_i = 1 if x_i belongs in class 1 or d_i = -1 if x_i belongs to class -1
    """
    def predict(self, x):
        # Not linear kernel
        if self.w is None:
            prediction = np.zeros(len(x))
            for i in range(len(x)):
                w = 0
                for a, sv_y, sv in zip(self.a, self.support_vector_y, self.support_vector):
                    # weight = Σ ai*yi*K(xi,xi_sv)
                    if self.kernel == RBF:
                        w += a * sv_y * self.kernel(x[i], sv, self.gamma)
                    else:
                        w += a * sv_y * self.kernel(x[i], sv)
                prediction[i] = w
            return np.sign(prediction + self.b)
        else: # Linear Kernel
            return np.sign(np.dot(x, self.w) + self.b)


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def RBF(x, y, gamma):
    return np.exp(-gamma*(np.linalg.norm(x-y)**2))


if __name__ == '__main__':
    # Simple Example
    x_train = np.array([[1,7],[2,8],[3,8], [5,1], [6,-1], [7,3]])
    y_train = np.array([-1,-1,-1,1,1,1])

    x_test = np.array([[7,3], [1,7]])
    y_test = np.array([1,-1])
    clf = SVM(linear_kernel, C=0.01, gamma=0)

    clf.train(x_train, y_train)
    y_predict = clf.predict(x_test)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    # d = DataProcessing
    # sp500 = d.store_data('SPY', start='2000-01-01', end='2020-12-31', interval='1d')
    # X_train, y_train, X_test, y_test = d.processing_data(sp500)
    # clf = SVM(linear_kernel, C=0.01, gamma=0)
    #
    # clf.train(X_train, y_train)
    # y_predict1 = clf.predict(X_test)
    # # print(np.sum(y_predict1 == 1))
    #
    # correct1 = np.sum(y_predict1 == y_test)
    # # print("%d out of %d predictions correct" % (correct1, len(y_predict1)))
    #
    # clf = SVM(RBF, C=0.001, gamma=3)
    #
    # clf.train(X_train, y_train)
    # y_predict = clf.predict(X_test)
    # # print(np.sum(y_predict == 1))
    #
    # correct = np.sum(y_predict == y_test)
    # print(np.sum(y_test == 1))
    #
    # print("Linear Kernel")
    # print("%d out of %d predictions correct" % (correct1, len(y_predict1)))
    # print("RBF kernel")
    # print("%d out of %d predictions correct" % (correct, len(y_predict)))

