import numpy as np
import matplotlib.pyplot as plt
import scipy
import Recursos.process_email as pe
import Recursos.get_vocab_dict as gvd
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm as sk

def show_contour(X, y, svm, name ='visualize boundary'):
    x1 = np.linspace(X[:, 0].min()-0.1, X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min()-0.1, X[:, 1].max(), 100)

    x1, x2 = np.meshgrid(x1, x2)    
    yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape) 

    pos = (y == 1).ravel()    
    neg = (y == 0).ravel()    

    plt.figure()    
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')    
    plt.scatter(X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')    
    plt.contour(x1, x2, yp)    
    plt.savefig('Práctica 6/Recursos/{}.png'.format(name), dpi=200)
    plt.close()

# 1.1. Kernel lineal
def lineal_kernel():
    data = loadmat('Práctica 6/Recursos/ex6data1.mat')

    X, Y = data['X'], data['y'].ravel()

    svm = sk.SVC(kernel='linear', C=1)
    svm.fit(X,Y)

    show_contour(X, Y, svm,'lineal_kernel_c_1')

    svm = sk.SVC(kernel='linear', C=100)
    svm.fit(X, Y)

    show_contour(X, Y, svm, 'lineal_kernel_c_100')

# 1.2. Kernel gaussiano
def gaussian_kernel():
    data = loadmat('Práctica 6/Recursos/ex6data2.mat')

    X, Y = data['X'], data['y'].ravel()

    sigma = 0.1

    svm = sk.SVC(kernel='rbf', C=1, gamma=1 / (2 * sigma ** 2))
    svm.fit(X, Y)

    show_contour(X, Y, svm, 'gaussian_kernel_c_1_s_0.1')

# 1.3. Elección de los parámetros C y sigma
def parameter_selection():
    data = loadmat('Práctica 6/Recursos/ex6data3.mat')

    X, Y = data['X'], data['y'].ravel()
    X_val, Y_val = data['Xval'], data['yval'].ravel()

    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    C_aux, sigma_aux = np.meshgrid(values, values)

    combinations = np.vstack((C_aux.ravel(), sigma_aux.ravel())).T
    errors = []

    for v in combinations:
        c = sk.SVC(kernel='rbf', C=v[0], gamma=1 / (2 * v[1]**2))
        c.fit(X, Y)

        errors.append(c.score(X_val, Y_val))
    
    best_value = combinations[np.argmax(errors)]

    best = sk.SVC(kernel='rbf', C=best_value[0], gamma=1 / (2 * best_value[1] ** 2))
    best.fit(X, Y)

    show_contour(X, Y, best, 'parameter_selection')

def support_vector_machines():
    lineal_kernel()
    gaussian_kernel()
    parameter_selection()


def main():
    support_vector_machines()

    return 0; 

main()
