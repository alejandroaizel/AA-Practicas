import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

def load_csv(file_name):
    values = read_csv(file_name, header=None).values

    return values.astype(float)

def show_data(X, Y, x_lable, y_lable, ad_legends, not_legend, name):
    admited = np.where(Y == 1)
    not_admited = np.where(Y == 0)

    ad = plt.scatter(X[admited, 0], X[admited, 1], marker='+', c='k')
    not_ad = plt.scatter(X[not_admited, 0], X[not_admited, 1], c='y')

    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.legend((ad, not_ad), (ad_legends, not_legend))
    plt.savefig('Práctica 2/Recursos/{}.png'.format(name), dpi=200)

def show_result(X, Y, Theta, x_lable, y_lable, ad_legends, not_legend, Lambda=None, poly=None, name='graph'):
    plt.figure()

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

    show_data(X, Y, x_lable, y_lable, ad_legends, not_legend, name)

    if Lambda == None:
        h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(Theta))
    else:
        plt.title('lambda = {}'.format(Lambda))

        h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(Theta))
    
    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    plt.savefig('Práctica 2/Recursos/{}_result.png'.format(name), dpi=200)
    plt.close()

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def gradient(Theta, XX, Y):
    H = sigmoid(np.matmul(XX, Theta))

    return (1 / len(Y)) * np.matmul(XX.T, H - Y)

def cost(Theta, X, Y):
    H = sigmoid(np.matmul(X, Theta))

    return (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))

def evaluation(X, Y, Theta):
    hipotesis = sigmoid(np.matmul(X, Theta))

    hipotesis[hipotesis >= 0.5] = 1
    hipotesis[hipotesis < 0.5] = 0

    result = hipotesis == Y
    
    return (np.sum(result) / len(result)) * 100

def logistic_regression():
    Data = load_csv('Práctica 2/Recursos/ex2data1.csv')

    X = Data[:, :-1]
    Y = Data[:, -1]

    X = np.hstack([np.ones([len(X), 1]), X])
    Theta = np.zeros(len(X[0]))

    result = opt.fmin_tnc(func=cost , x0=Theta, fprime=gradient , args=(X, Y))
    theta_opt = result [0]

    show_result(X[:, 1:], Y, theta_opt, 'Exam 1 score', 'Exam 2 score', 'Admited', 'Not Admited')

    print('Correctly classified: {} %'.format(evaluation(X, Y, theta_opt)))


def gradient_reg(Theta, XX, Y, Landa):
    G = gradient(Theta, XX, Y)

    G[1:] += Landa / len(XX) * Theta[1:]

    return G

def cost_reg(Theta, X, Y, Landa):
    C = cost(Theta, X, Y)

    return C + Landa / (2 * len(X)) * np.sum(Theta ** 2)

def regularized_regression():
    Data = load_csv('Práctica 2/Recursos/ex2data2.csv')

    X = Data[:, :-1]
    Y = Data[:, -1]

    poly = PolynomialFeatures(6)
    X_n = poly.fit_transform(X)

    Theta = np.zeros(len(X_n[0]))
    Lambda = 20

    result = opt.fmin_tnc(func=cost_reg, x0=Theta, fprime=gradient_reg, args=(X_n, Y, Lambda))
    theta_opt = result[0]
    
    show_result(X, Y, theta_opt, 'Microchip test 1', 'Microchip test 2', 'y = 1', 'y = 0', Lambda, poly, name='reg_graph')

    print('Correctly classified: {} %'.format(evaluation(X_n, Y, theta_opt)))


def main():
    logistic_regression()
    regularized_regression()


main()
