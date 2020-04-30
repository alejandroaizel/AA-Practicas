from scipy . io import loadmat

import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def gradient(Theta, XX, Y):
    H = sigmoid(np.matmul(XX, Theta))

    return (1 / len(Y)) * np.matmul(XX.T, H - Y)

def gradient_reg(Theta, XX, Y, Landa):
    G = gradient(Theta, XX, Y)

    G[1:] += Landa / len(XX) * Theta[1:]

    return G

def cost(Theta, X, Y):
    H = sigmoid(np.matmul(X, Theta))

    return (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))

def cost_reg(Theta, X, Y, Landa):
    C = cost(Theta, X, Y)

    return C + Landa / (2 * len(X)) * np.sum(Theta ** 2)

def oneVsAll (X, y, num_labels, reg):
    Thetas = np.zeros((num_labels, len(X[0])))

    for i in range(num_labels):
        Thetas[i] = opt.fmin_tnc(func=cost_reg, x0=Thetas[i], fprime=gradient_reg, args=(X, [i] == y, reg))[0]

    return Thetas

def evaluation (X, y, Thetas):
    h = sigmoid(np.dot(Thetas, np.transpose(X)))

    maxValues = np.argmax(h, axis=0)

    result = maxValues == y

    print("Se han clasificado correctamente: {} %".format(np.sum(result) / len(result) * 100))

def multi_class(X, y):
    y[y==10] = 0

    Thetas = oneVsAll(X, y, 10, 0.5)

    evaluation(X, y, Thetas)


def neural_network(X, y):
    weights = loadmat('Práctica 3/Recursos/ex3weights.mat') 

    theta1, theta2 = weights['Theta1'], weights['Theta2']

    hidden_layer = sigmoid(np.matmul(theta1, np.transpose(X)))
    hidden_layer = np.vstack([np.ones([len(X)]), hidden_layer])

    output_layer = sigmoid(np.matmul(theta2, hidden_layer))

    max_values = np.argmax(output_layer, axis=0) + 1

    correct_values = y == max_values

    print("Se han clasificado correctamente: {} %".format(np.sum(correct_values) / len(correct_values) * 100))

def main():
    data = loadmat('Práctica 3/Recursos/ex3data1.mat')

    y = np.squeeze(data['y'])
    X = data['X']

    X = np.hstack([np.ones([len(X), 1]), X])
    
    multi_class(X, y)
    neural_network(X, y)


main()
