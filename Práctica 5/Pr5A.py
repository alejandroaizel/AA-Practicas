import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
from scipy.optimize import minimize


def gradient(Theta, X, Y):
    H = np.matmul(X, Theta)

    return np.matmul(np.transpose(X), H - Y) / len(Y)

def cost(Theta, X, Y):
    H = np.dot(X, Theta)

    aux = (H - Y) ** 2

    return aux.sum() / (2 * len(X))

def gradient_reg(Theta, X, Y, Landa):
    G = gradient(Theta, X, Y)

    G[1:] += Landa / len(X) * Theta[1:]

    return G

def cost_reg(Theta, X, Y, Landa):
    C = cost(Theta, X, Y)

    aux2= np.sum(Theta[1:] ** 2)
    aux1 = (2 * len(X))

    aux = aux1 * aux2

    return C + Landa / aux

def cost_gradient_reg(Theta, X, Y, Landa):
    return [cost_reg(Theta, X, Y, Landa), gradient_reg(Theta, X, Y, Landa)]

def polynomial_features(X, p):
    for i in range(2, p + 1):
        aux = X[:, 0] ** i 

        X = np.hstack((X, np.reshape(aux, (aux.shape[0], 1))))

    return X

def normalize(X):
    Mu = np.mean(X, axis=0)
    Sigma = np.std(X,axis=0)
    X_norm = (X - Mu) / Sigma

    return [X_norm, Mu, Sigma]

def test_hypotesis(norm_X, Y, X_test, Y_test, Mu, Sigma, Landa, p):
    new_X_test = polynomial_features(X_test, p)
    norm_X_test = (new_X_test - Mu) / Sigma
    norm_X_test = np.hstack((np.ones((len(norm_X_test), 1)), norm_X_test))

    Theta = np.ones(len(norm_X[0]))

    fmin = minimize(fun=cost_gradient_reg, x0=Theta, args=(norm_X, Y, Landa), method='TNC', jac=True)

    cost = cost_reg(fmin.x, norm_X_test, Y_test, Landa)

    print("Con λ = {}, el coste de X_text es: {}".format(Landa, cost))

def linear_regression_graph(X, Y, Theta, name):
    plt.figure()
    
    X_l = [np.amin(X), np.amax(X)]
    Y_l = [Theta[0] + Theta[1] * X_l[0], Theta[0] + Theta[1] * X_l[1]]

    plt.plot(X, Y, 'x', c='r')
    plt.plot(X_l, Y_l, c='b')
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.savefig('Práctica 5/Recursos/{}.png'.format(name), dpi=200)

def learning_curve_graph(X1, X2, x_label, title, name):
    plt.figure()

    Y = range(len(X1))

    plt.title(title)
    plt.plot(Y, X1, c='orange', label="Train")
    plt.plot(Y, X2, c='blue', label="Cross Validation")
    plt.xlabel(x_label)
    plt.ylabel("Error")
    plt.legend()
    plt.savefig('Práctica 5/Recursos/{}.png'.format(name), dpi=200)

def polynomial_regression_graph(X, Y, Theta, Mu, Sigma, p, title, name):
    plt.figure()

    X_t = np.arange(np.amin(X) - 5, np.amax(X) + 5, 0.05)
    X_t = X_t.reshape(-1, 1)
    X_p = polynomial_features(X_t, p)
    X_p = (X_p - Mu) / Sigma
    X_p = np.hstack((np.ones((len(X_p), 1)), X_p))
    Y_t = np.dot(X_p, Theta)

    plt.title(title)
    plt.plot(X, Y, 'x', c='r')
    plt.plot(X_t, Y_t, c='b')
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.savefig('Práctica 5/Recursos/{}.png'.format(name), dpi=200)

def learning_curve_graph2(X, Y1, Y2, x_label, title, name):
    plt.figure()
    plt.title(title)
    plt.plot(X, Y1, c='orange', label="Train")
    plt.plot(X, Y2, c='blue', label="Cross Validation")
    plt.xlabel(x_label)
    plt.ylabel("Error")
    plt.legend()
    plt.savefig('Práctica 5/Recursos/{}.png'.format(name), dpi=200)


# 1. Regresión Lineal Regularizada
def linear_regression(X, Y):
    X = np.hstack((np.ones((len(X), 1)), X))

    Theta = np.ones(2)
    Landa = 1

    cost, grad = cost_gradient_reg(Theta, X, Y, Landa)

    print("For Theta = {} and Lamda = {}:\n  Cost = {}\n  Gradient = {}".format(Theta, Landa, cost, grad))

    Landa = 0

    fmin = minimize(fun=cost_gradient_reg, x0=Theta, args=(X, Y, Landa), method='TNC', jac=True)

    linear_regression_graph(X[:,1:], Y, fmin.x, name="linear_regression")


# 2. Curva de Aprendizaje
def learning_curve(X, Y, X_val, Y_val, Landa, name="learning_curves"):
    m = len(X)

    X = np.hstack((np.ones((len(X), 1)), X))
    X_val = np.hstack((np.ones((len(X_val), 1)), X_val))

    Theta = np.ones(len(X[0]))

    cost = []
    cost_val = []

    for i in range(1, m + 1):
        fmin = minimize(fun=cost_gradient_reg, x0=Theta, args=(X[0:i], Y[0:i], Landa), method='TNC', jac=True)

        cost.append(fmin['fun'])
        cost_val.append(cost_gradient_reg(fmin.x, X_val, Y_val, Landa)[0])
    
    learning_curve_graph(cost, cost_val, "Number of training examples", "Learning curve for lineal regression (λ = {})".format(Landa), name=name)


# 3. Regresión Polinomial
def polinomial_regression(X, Y, X_val, Y_val, Landa, p):
    new_X = polynomial_features(X, p)
    norm_X, mu_X, sigma_X = normalize(new_X)
    norm_X = np.hstack((np.ones((len(norm_X), 1)), norm_X))

    Theta = np.ones(len(norm_X[0]))

    fmin = minimize(fun=cost_gradient_reg, x0=Theta, args=(norm_X, Y, Landa), method='TNC', jac=True)

    polynomial_regression_graph(X, Y, fmin.x, mu_X, sigma_X, p, "Learning curve for lineal regression (λ = {})".format(Landa), "polinomial_regression_graph")

    new_X_val = polynomial_features(X_val, p)
    norm_X_val = (new_X_val - mu_X) / sigma_X

    learning_curve(norm_X[:,1:], Y, norm_X_val, Y_val, Landa=0, name="learning_curves_2_λ0")
    learning_curve(norm_X[:,1:], Y, norm_X_val, Y_val, Landa=1, name="learning_curves_2_λ1")
    learning_curve(norm_X[:,1:], Y, norm_X_val, Y_val, Landa=100, name="learning_curves_2_λ100")


# 4. Selección del Parámetro λ
def parameter_selection(X, Y, X_val, Y_val, X_test, Y_test, p):
    Landas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    cost = []
    cost_val = []

    new_X = polynomial_features(X, p)
    norm_X, Mu, Sigma = normalize(new_X)
    norm_X = np.hstack((np.ones((len(norm_X), 1)), norm_X))

    new_X_val = polynomial_features(X_val, p)
    norm_X_val = (new_X_val - Mu) / Sigma
    norm_X_val = np.hstack((np.ones((len(norm_X_val), 1)), norm_X_val))

    Theta = np.ones(len(norm_X[0]))

    for i in Landas:
        fmin = minimize(fun=cost_gradient_reg, x0=Theta, args=(norm_X, Y, i), method='TNC', jac=True)

        cost.append(fmin['fun'])
        cost_val.append(cost_reg(fmin.x, norm_X_val, Y_val, i))
    
    learning_curve_graph2(Landas, cost, cost_val, "Lambda", "Selecting λ using a cross validation set", "selecting_lambda")

    test_hypotesis(norm_X, Y, X_test, Y_test, Mu, Sigma, 3, p)

    
def main():
    data = loadmat('Práctica 5/Recursos/ex5data1.mat')

    X, Y = data['X'], data['y'].ravel()
    X_val, Y_val = data['Xval'], data['yval'].ravel()
    X_test, Y_test = data['Xtest'], data['ytest'].ravel()
    Landa, p = 0, 8

    linear_regression(X, Y)
    learning_curve(X, Y, X_val, Y_val, Landa)
    polinomial_regression(X, Y, X_val, Y_val, Landa, p)
    parameter_selection(X, Y, X_val, Y_val, X_test, Y_test, p)
    

main()