import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.1, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.w_list = []
        self.b_list = []


    def fit(self, x, y):
        self.w = np.zeros(x.shape[1])
        self.b = 0

        self.w_list = [list(self.w)]
        self.b_list = [self.b]
        for _ in range(self.n_iters):
            for i in range(x.shape[0]):
                if y[i] * (np.dot(x[i], self.w) + self.b) >= 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x[i], y[i]))
                    self.b -= self.lr * (-y[i])

            if np.linalg.norm(self.w - self.w_list[-1]) < 1e-5:
                break
            else:
                self.w_list.append(list(self.w))
                self.b_list.append(self.b)


if __name__ == '__main__':
    # Construct the data
    data = np.array([[4, 3], [4, 8], [7, 2], [-1, -2], [-1, 3], [2, -1], [2, 1]])
    target = np.array([1, 1, 1, -1, -1, -1, -1])
    
    # Train the model
    svm = LinearSVM(n_iters=10000)
    svm.fit(data, target)
    print('w', svm.w)
    print('b', svm.b)
    plt.plot(np.linalg.norm(svm.w_list, axis=1), label='weight')
    plt.plot(svm.b_list, label='bias')
    plt.legend()
    plt.title('Weight and Bias Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.savefig('hw5-p3-a.png', dpi=300, bbox_inches='tight')
    plt.show()

    # For each data point, calculate the distance to the hyperplane
    distance = np.abs(np.dot(data, svm.w) + svm.b) / np.linalg.norm(svm.w)
    min_index = np.argsort(distance)
    print('distance', distance)

    # Plot the data points and the hyperplane
    x_axis = np.arange(-5, 11)
    y_axis = -1 * (x_axis * svm.w[0] + svm.b) / svm.w[1]
    margin = -1 / svm.w[1]
    plt.figure(figsize=(6, 7))
    plt.scatter(data[:3, 0], data[:3, 1], label='class 1', c='steelblue')
    plt.scatter(data[3:, 0], data[3:, 1], label='class -1', c='tan')
    plt.plot(x_axis, y_axis, label='hyperplane', c='lightseagreen')
    plt.plot(x_axis, y_axis + margin, label='margin', c='gray')
    plt.plot(x_axis, y_axis - margin, c='gray')
    plt.scatter(data[min_index[:2], 0], data[min_index[:2], 1], label='support vector', s=120, facecolors='none', edgecolors='r')
    plt.legend()
    plt.title('Linear SVM')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('hw5-p3-b.png', dpi=300, bbox_inches='tight')
    plt.show()

    

