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
    Mu = np.mean(X[:,1:], axis=0)
    Sigma = np.std(X[:,1:],axis=0)
    X_norm = (X[:,1:] - Mu) / Sigma
    X_norm = np.hstack((np.ones((len(X), 1)), X_norm))

    return [X_norm, Mu, Sigma]

def normalize_val_or_test(X,mu,sigma):
    X_norm = (X[:,1:] - mu) / sigma
    X_norm = np.hstack((np.ones((len(X), 1)), X_norm))
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


def show_graph_normalized(X, Y,X_norm,theta, name = 'graph_normalized'):

    Y_l = X_norm*theta

    plt.figure()
    plt.scatter(X.T[1], Y, c='r', marker='x', s=40)
    #plt.plot(X, Y_l, c='b')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.savefig('Práctica 5/Recursos/{}.png'.format(name), dpi=200)

def show_learning_curve(Train, val,num_examples, name = 'Learning curve'):


    plt.figure()
    plt.plot(np.arange(0,num_examples), Train, c='b', label = 'Train')
    plt.plot(np.arange(0,num_examples), val, c='y', label = 'Error')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('Práctica 5/Recursos/{}.png'.format(name), dpi=200)

def learning_curve(Theta, X, y, Xval, yval, landa):

    train_error = np.zeros(len(X))
    val_error = np.zeros(len(X))
    for i in range(0,len(X)):
        fmin = minimize(fun=error,x0=Theta,args=(X[0:i],y[0:i],landa ),method='TNC',jac=False) 
        theta = fmin['x']
        train_error[i] = error(theta, X[:i],y[:i],landa)
        val_error[i] = error(theta, Xval,yval,landa) 

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
    Landa = 0
    fmin = minimize(fun=cost_and_gradient,x0=Theta,args=(X,y,Landa),method='TNC',jac=True)
    show_graph(X,y,fmin['x'])

    error_train, error_val = learning_curve(Theta,X,y,Xval,yval,0)
    show_learning_curve(error_train,error_val, len(X))

    

    X_featurized = add_features(X,8)
    Xval_featurized = add_features(Xval,8)
    Xtest_featurized = add_features(Xtest,8)

    X_normalized, mu, sigma = normalize(X_featurized)
    Xval_normalized = normalize_val_or_test(Xval_featurized,mu,sigma)
    Xtest_normalized = normalize_val_or_test(Xtest_featurized,mu,sigma)
    Theta_featurized= np.ones(9)

    # TODO 
    fmin_norm =  minimize(fun=cost_and_gradient,x0=Theta_featurized,args=(X_normalized,y,0),method='TNC',jac=True)
    show_graph_normalized(X,y,X_normalized,fmin_norm['x'],name='Graph normalized')

    error_train, error_val = learning_curve(Theta_featurized,X_normalized,y,Xval_normalized,yval,0)
    show_learning_curve(error_train,error_val, len(X), name = 'Learning curve for polynomial hypotesis')
    # TODO lo que esta entre los todos no va del todo
    
    Landas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    errors_train = []
    errors_val = []
    for landa in Landas:
        res = minimize(fun=cost_and_gradient,x0=Theta_featurized,args=(X_normalized,y,landa),method='TNC', jac=True);
        theta = res['x']
        errors_train.append(error(theta, X_normalized, y, landa))
        errors_val.append(error(theta, Xval_normalized, yval, landa))
    
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