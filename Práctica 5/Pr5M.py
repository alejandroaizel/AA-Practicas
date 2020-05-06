import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import scipy.optimize as opt

def cost_and_gradient(Theta, XX, Y, Landa):
    

    cost = 1 / 2 / len(XX) * np.sum((Theta.dot(XX.T) - Y)**2) + Landa / 2 / len(XX) * np.sum(Theta[1:]**2) 

    gradient = np.transpose(XX).dot((np.dot(XX,np.transpose(Theta)) - Y)) / len(XX) + Landa / len(XX) * Theta[1:]
    

    return cost,gradient

def cost_reg(Theta, XX, Y, Landa):

    dif = np.sum((Theta.dot(XX.T) - Y)**2)
    cost = (1 / 2 * len(XX)) * dif 
    return cost
def error(Theta,XX,y,landa):   
    H = np.dot(XX, Theta)
    aux = (H - y) ** 2
    
    return aux.sum() / (2 * len(XX))

def show_graph(X, Y, Theta, name='graph'):

    X_l = [np.amin(X), np.amax(X)]
    Y_l = [Theta[0] + Theta[1] * X_l[0], Theta[0] + Theta[1] * X_l[1]]

    plt.figure()
    plt.scatter(X.T[1], Y, c='r', marker='x', s=40)
    plt.plot(X_l, Y_l, c='b')
    plt.xlabel('Población de la ciudad en 10.000s')
    plt.ylabel('Ingresos en $10.000s')
    plt.savefig('Práctica 5/Recursos/{}.png'.format(name), dpi=200)

def show_learning_curve(Train, val,num_examples):


    plt.figure()
    plt.plot(np.arange(0,num_examples), Train, c='b', label = 'Train')
    plt.plot(np.arange(0,num_examples), val, c='y', label = 'Error')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('Práctica 5/Recursos/{}.png'.format('Learning Curve'), dpi=200)

def learning_curve(Theta, X, y, Xval, yval, landa):

    train_error = np.zeros(len(X))
    val_error = np.zeros(len(X))
    for i in range(0,len(X)):
        fmin = minimize(fun=error,x0=Theta,args=(X[0:i],y[0:i],landa ),method='TNC',jac=False) 
        theta = fmin['x']
        train_error[i] = cost_reg(theta, X[:i],y[:i],landa)
        val_error[i] = cost_reg(theta, Xval,yval,landa) 

    return train_error, val_error

def main():
    data = loadmat('Práctica 5/Recursos/ex5data1.mat')

    X, y = data['X'], data['y'].ravel()
    Xval, yval = data['Xval'], data['yval'].ravel()
    Xtest, ytest = data['Xtest'], data['ytest'].ravel()


    X = np.hstack((np.ones((len(X), 1)), X))
    Xval = np.hstack((np.ones((len(Xval), 1)), Xval))
    Xtest = np.hstack((np.ones((len(Xtest), 1)), Xtest))  


    Theta = np.ones(2)
    Landa = 1
    
    fmin =  minimize(fun=cost_and_gradient,x0=Theta,args=(X,y,Landa),method='TNC',jac=True)
    show_graph(X,y,fmin['x'])

    error_train, error_val = learning_curve(Theta,X,y,Xval,yval,0)
    show_learning_curve(error_train,error_val, len(X))



main()