import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import scipy.optimize as opt
from scipy.optimize import minimize
from sklearn import svm as sk



def load_csv(file_name):
    values = read_csv(file_name, header=None).values

    return values[1:,:].astype(float)

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

def normalize(X):
    Mu = np.mean(X, axis=0)
    Sigma = np.std(X,axis=0)
    X_norm = (X - Mu) / Sigma

    return [X_norm, Mu, Sigma]

def visualize_boundary(X, y, svm, name ='visualize boundary'):
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
    plt.savefig('Práctica Final/Recursos/{}.png'.format(name), dpi=200)
    plt.close()

def normal_ecuation_method(X, Y):
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)


#1
def lineal_regresion(X_test,Y_test, X_val, Y_val):

    X_norm ,mu, sigma = normalize(X_test)
    X_val = (X_val-mu)/sigma
    X_norm = np.hstack((np.ones((len(X_norm), 1)), X_norm))
    X_val = np.hstack((np.ones((len(X_val), 1)), X_val))

    theta = np.zeros(len(X_norm[0]))
    Lambda = 0

    fmin = minimize(fun=cost_gradient_reg, x0=theta, args=(X_norm, Y_test, Lambda), method='TNC', jac=True)

    correct_values = (Y_val == np.dot(X_val, fmin.x))

    print("Se han clasificado correctamente: {} %".format(np.sum(correct_values) / len(correct_values) * 100))


#2
def logistic_regresion(X_test,Y_test, X_val, Y_val):
    print()

#3
def neural_network(X_test,Y_test, X_val, Y_val):
    print()

#4
def SVM(X_test,Y_test, X_val, Y_val):


    svm = sk.SVC(kernel='linear', C=1)
    svm.fit(X_test,Y_test)

    accuracy = svm.score(X_val, Y_val)

    print("Se han clasificado correctamente: {} %".format(accuracy*100))

    svm = sk.SVC(kernel='linear', C=100)
    svm.fit(X_test,Y_test)

    accuracy = svm.score(X_val, Y_val)

    print("Se han clasificado correctamente: {} %".format(accuracy*100))



def split_data(Data):

    X_blue = Data[:, 2:21]
    Y_blue = Data[:, 1]

    X_red = Data[:, 21:41]
    Y_red = np.ones(Y_blue.size) - Y_blue

    X = np.vstack((X_blue,X_red))
    Y = np.hstack((Y_blue,Y_red))

    total_examples, features = X.shape

    test_size = int(0.7 * total_examples)
    val_size= total_examples-test_size

    X_test = X[0:test_size, :]
    X_val = X[test_size+1:total_examples, :]

    Y_test = Y[0:test_size] 
    Y_val = Y[test_size+1:total_examples]

    return X_test, X_val, Y_test, Y_val

def main():
    

    Data = load_csv('Práctica Final/Recursos/high_diamond_ranked_10min.csv')

    X_test, X_val, Y_test, Y_val = split_data(Data)

    
    lineal_regresion(X_test,Y_test, X_val, Y_val)
    logistic_regresion(X_test,Y_test, X_val, Y_val)
    neural_network(X_test,Y_test, X_val, Y_val)
    SVM(X_test,Y_test, X_val, Y_val)




main()