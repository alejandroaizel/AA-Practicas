import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def load_csv(file_name):
    values = read_csv(file_name, header=None).values

    return values.astype(float)

def cost(X, Y, Theta):
    H = np.dot(X, Theta)

    aux = (H - Y) ** 2

    return aux.sum() / (2 * len(X))

def make_data(t0_range, t1_range, X, Y):
    step = 0.1
    Theta0 = np.arange(t0_range[0], t0_range[1], step)
    Theta1 = np.arange(t1_range[0], t1_range[1], step)

    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)

    Cost = np.empty_like(Theta0)

    for ix, iy in np.ndindex(Theta0.shape):
        Cost[ix, iy] = cost(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])

    return [Theta0, Theta1, Cost]

def show_contour_cost(X, Y, Z, Theta, name='contour_cost'):
    plt.figure()
    plt.contour(X, Y, Z, np.logspace(-2, 3, 20), cmap=cm.coolwarm)
    plt.plot(Theta[0], Theta[1], 'x', c='r')
    plt.xlabel('θ₀')
    plt.ylabel('θ₁')
    plt.savefig('Práctica 1/Recursos/{}.png'.format(name), dpi=200)

def show_3d_costs(X, Y, Z, name='3d_cost'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.view_init(elev=15, azim=230)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.xlabel('θ₀')
    plt.ylabel('θ₁')
    plt.locator_params(axis='x', nbins=5)
    plt.savefig('Práctica 1/Recursos/{}.png'.format(name), dpi=200)

def show_2d_costs(Costs, name='cost'):
    X = np.linspace(0, len(Costs), len(Costs))

    plt.figure()
    plt.plot(X, Costs, '.', c='r')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.savefig('Práctica 1/Recursos/{}.png'.format(name), dpi=200)

def show_costs(X, Y, Theta, Costs):
    t0_range, t1_range = [-10, 10], [-1, 4]

    X, Y, Z = make_data(t0_range, t1_range, X, Y)

    show_2d_costs(Costs)
    show_3d_costs(X, Y, Z)
    show_contour_cost(X, Y, Z, Theta)

def show_graph(X, Y, Theta, name='graph'):
    X_l = [np.amin(X), np.amax(X)]
    Y_l = [Theta[0] + Theta[1] * X_l[0], Theta[0] + Theta[1] * X_l[1]]

    plt.figure()
    plt.plot(X, Y, 'x', c='r')
    plt.plot(X_l, Y_l, c='b')
    plt.xlabel('Población de la ciudad en 10.000s')
    plt.ylabel('Ingresos en $10.000s')
    plt.savefig('Práctica 1/Recursos/{}.png'.format(name), dpi=200)

def gradient_descent(X, Y, alpha=0.3):
    theta0, theta1, Thetas, Costs = 0, 0, [], []

    while len(Costs) < 2 or Costs[len(Costs) - 2] - Costs[len(Costs) - 1] > 0.00001:
        Theta = [theta0, theta1]

        theta0 = theta0 - alpha / len(X) * (np.dot(X, Theta) - Y).sum()
        theta1 = theta1 - alpha / len(X) * ((np.dot(X, Theta) - Y) * np.squeeze(X[:, 1:])).sum()

        Thetas.append([theta0, theta1])
        Costs.append(cost(X, Y, [theta0, theta1]))

    return [Thetas[len(Thetas) - 1], Costs]

def one_var_regression():
    Data = load_csv('Práctica 1/Recursos/ex1data1.csv')

    X = Data[:, :-1]
    Y = Data[:, -1]
    X = np.hstack([np.ones([len(X), 1]), X])

    Theta, Costs = gradient_descent(X, Y, alpha=0.01)

    show_graph(X[:,1:], Y, Theta)
    show_costs(X, Y, Theta, Costs)


def check_ex(Th_grad, Th_norm, Mu, Sigma, X):
    X_norm = X.copy()
    X_norm[1:] = (X_norm[1:] - Mu) / Sigma

    print('Gradient descent result: {}'.format(np.dot(X_norm, Th_grad)))
    print('Normalized funcn result: {}'.format(np.dot(X, Th_norm)))

def normalize(X):
    Mu = np.mean(X, axis=0)
    Sigma = np.std(X,axis=0)
    X_norm = (X - Mu) / Sigma

    return [X_norm, Mu, Sigma]

def gradient_descent_mult(X, Y, alpha=0.3):
    Theta, Costs, m = np.zeros(len(X[0])), [], len(X)

    for i in range(1500):

        Aux = Theta.copy()

        Theta = Aux - alpha / m * (np.dot((np.dot(X, Aux) - Y), X))

        Costs.append(cost(X, Y, Theta))

    return [Theta, Costs]

def normal_ecuation_method(X, Y):
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)

def show_diff_alpha(X, Y):
    Alphas = np.array([0.3, 0.1, 0.03, 0.01])
    Colors = np.array(['salmon', 'gold', 'violet', 'aquamarine'])

    plt.figure()

    for i in range(4):
        Theta, Cost = gradient_descent_mult(X, Y, Alphas[i])

        X_plot = np.linspace(0, len(Cost), len(Cost))

        plt.plot(X_plot, Cost, '.', c=Colors[i], label='α: {}'.format(Alphas[i]))
    
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.legend()
    plt.savefig('Práctica 1/Recursos/mul_alphas.png', dpi=200)

def mult_var_regression():
    Data = load_csv('Práctica 1/Recursos/ex1data2.csv')
    
    X = Data[:, :-1]
    Y = Data[:, -1]
    X = np.hstack([np.ones([len(X), 1]), X])

    X_norm, Mu, Sigma = normalize(X[:, 1:])
    X_norm = np.hstack([np.ones([len(X_norm), 1]), X_norm])
    
    Th_grad, Cost = gradient_descent_mult(X_norm, Y, alpha=0.1)
    Th_norm = normal_ecuation_method(X, Y)

    check_ex(Th_grad, Th_norm, Mu, Sigma, [1.0, 1650, 3])
    show_2d_costs(Cost, name='mult_cost')
    show_diff_alpha(X_norm, Y)


def main():
    #one_var_regression()
    mult_var_regression()


main()
