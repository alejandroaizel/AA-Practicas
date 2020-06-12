import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import scipy.optimize as opt

def cost_and_gradient(Theta, XX, Y, Landa):


    cost = 1 / 2 / len(XX) * np.sum((Theta.dot(XX.T) - Y)**2) + Landa / 2 / len(XX) * np.sum(Theta[1:]**2) 
    regul =  Landa / len(XX) * Theta[1:]
    gradient =  np.transpose(XX).dot((np.dot(XX,np.transpose(Theta)) - Y)) / len(XX)

    gradient[1:] += regul

    return cost,gradient

def cost_reg(Theta, XX, Y, Landa):

    dif = np.sum((Theta.dot(XX.T) - Y)**2)
    cost = (1 / 2 * len(XX)) * dif 
    return cost
def error(Theta,XX,y,landa):   
    H = np.dot(XX, Theta)
    aux = (H - y) ** 2
    
    return aux.sum() / (2 * len(XX))

def add_features(X, p):
    for i in range(2,p+1):
       x1 = X[:,1]**i 
       X = np.hstack((X, np.reshape(x1,(x1.shape[0],1))))
    return X

def normalize(X):
    X_norm = np.zeros_like(X)
    X_norm[:,0] = X[:,0]
    for i in range(1,X.shape[1]):
        mu = np.mean(X[:,i])
        sigma = np.std(X[:,i])
        data = (X[:,i]-mu) / sigma
        X_norm[:,i] = data
    return X_norm, mu, sigma

def normalize_val_or_test(X,mu,sigma):
    X_norm = np.zeros_like(X)
    X_norm[:,0] = X[:,0]
    for i in range(1,X.shape[1]):
        data = (X[:,i]-mu) / sigma
        X_norm[:,i] = data

    return X_norm
def show_graph(X, Y, Theta, name='graph'):

    X_l = [np.amin(X), np.amax(X)]
    Y_l = [Theta[0] + Theta[1] * X_l[0], Theta[0] + Theta[1] * X_l[1]]

    plt.figure()
    plt.scatter(X.T[1], Y, c='r', marker='x', s=40)
    plt.plot(X_l, Y_l, c='b')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.savefig('Práctica 5/Recursos/{}.png'.format(name), dpi=200)


def show_graph_normalized(X, Y, theta, mu , sigma, name = 'graph_normalized'):

    x_pts = np.linspace(-60, 40, 12)
    y_pts = theta[0] * np.ones(12)
    for i in range(1,9):
        y_pts += theta[i] * (x_pts**i - mu[i-1]) / sigma[i-1]


    plt.figure()
    plt.plot(X[1], Y, 'rx', ms=8)
    plt.plot(x_pts, y_pts, 'b--')
    plt.xlim(-60,50)
    plt.ylim(-10,60)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
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
    #fmin = minimize(fun=cost_and_gradient,x0=Theta,args=(X,y,Landa),method='TNC',jac=True)
    #show_graph(X,y,fmin['x'])

    #error_train, error_val = learning_curve(Theta,X,y,Xval,yval,0)
    #show_learning_curve(error_train,error_val, len(X))

    X_featurized = add_features(X,8)
    Xval_featurized = add_features(Xval,8)
    Xtest_featurized = add_features(Xtest,8)

    X_normalized, mu, sigma = normalize(X_featurized)
    Xval_normalized = normalize_val_or_test(Xval_featurized,mu,sigma)
    Xtest_normalized = normalize_val_or_test(Xtest_featurized,mu,sigma)
    Theta_featurized= np.ones(9)

    #fmin_norm =  minimize(fun=cost_and_gradient,x0=Theta_featurized,args=(X_normalized,y,0),method='TNC',jac=True)

    #show_graph_normalized(X,y, fmin_norm['x'],mu, sigma)

    Landas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    errors_train = []
    errors_val = []
    for landa in Landas:
        res = minimize(fun=cost_and_gradient,x0=Theta_featurized,args=(X_normalized,y,landa),method='TNC', jac=True);
        theta = res['x']
        errors_train.append(cost_reg(theta, X_normalized, y, landa))
        errors_val.append(cost_reg(theta, Xval_normalized, yval, landa))
    
    plt.figure()
    plt.plot(Landas, errors_train,'b', label='Train')
    plt.plot(Landas, errors_val,'g', label='Cross validation')
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.xlim(0, 10)
    plt.ylim(0, 20)
    plt.legend(numpoints=1, loc=9)
    plt.savefig('Práctica 5/Recursos/{}.png'.format('Selecting Lambda'), dpi=200)



main()