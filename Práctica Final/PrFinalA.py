import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy.optimize as opt

def load_csv(file_name): # DONE
    values = read_csv(file_name, header=None).values

    return values[1:,:].astype(float)

def prepare_data(): # DONE
    Data = load_csv('Práctica Final/Recursos/high_diamond_ranked.csv')

    blue_team_X = Data[:, 2:21]
    red_team_X = Data[:, 21:41]

    blue_team_Y = Data[:, 1]
    red_team_Y = np.ones(blue_team_Y.size) - blue_team_Y

    X = np.concatenate((blue_team_X, red_team_X), axis=0)
    Y = np.concatenate([blue_team_Y, red_team_Y])

    return X, Y

def normalize(X):
    Mu = np.mean(X, axis=0)
    Sigma = np.std(X, axis=0)
    X_norm = (X - Mu) / Sigma

    return [X_norm, Mu, Sigma]


def cost(Theta, X, Y):
    H = sigmoid(np.matmul(X, Theta))

    return (- 1 / (len(X))) * (np.dot(Y, np.log(H)) + np.dot((1 - Y), np.log(1 - H)))

def cost_reg(Theta, X, Y, Landa):
    C = cost(Theta, X, Y)

    return C + Landa / (2 * len(X)) * np.sum(Theta ** 2)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def gradient(Theta, XX, Y):
    H = sigmoid(np.matmul(XX, Theta))

    return (1 / len(Y)) * np.matmul(XX.T, H - Y)





def gradient_descent_mult(X, Y, alpha=0.3):
    Theta, Costs, m = np.zeros(len(X[0])), [], len(X)

    for i in range(1500):

        Aux = Theta.copy()

        Theta = Aux - alpha / m * (np.dot((np.dot(X, Aux) - Y), X))

        Costs.append(cost(X, Y, Theta))

    return [Theta, Costs]

def normal_ecuation_method(X, Y):
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)


# 1. Regresión Lineal
def lineal_reggression(X, Y):
    X = np.hstack([np.ones([len(X), 1]), X])

    X_norm, Mu, Sigma = normalize(X[:, 1:])
    X_norm = np.hstack([np.ones([len(X_norm), 1]), X_norm])
    
    Th_grad, Cost = gradient_descent_mult(X_norm, Y, alpha=0.1)
    Th_norm = normal_ecuation_method(X, Y)

    check_ex(Th_grad, Th_norm, Mu, Sigma, [1.0, 1650, 3])
    show_2d_costs(Cost, name='mult_cost')
    show_diff_alpha(X_norm, Y)


# 2. Regresión Logística
def logistic_regression(X, Y):
    X = np.hstack([np.ones([len(X), 1]), X])

    Theta = np.zeros(len(X[0]))
    Lambda = 20

    result = opt.fmin_tnc(func=cost_reg, x0=Theta, fprime=gradient_reg, args=(X_n, Y, Lambda))
    theta_opt = result[0]
    
    print('Correctly classified: {} %'.format(evaluation(X_n, Y, theta_opt)))


# 3. Redes Neuronales
def neural_network(X, Y):
    a = 0


# 4. SVM
def support_vector_machines(X, Y):
    a = 0


def main():
    X, Y = prepare_data()

    lineal_reggression(X, Y)
    logistic_regression(X, Y)
    neural_network(X, Y)
    support_vector_machines(X, Y)


main()