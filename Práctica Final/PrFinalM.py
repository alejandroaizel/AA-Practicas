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

def cost_netword(output_layer, y, M):
    return 1 / M * np.trace((np.dot(-y, np.log(output_layer)) - np.dot((1 - y), np.log(1 - output_layer))))

def cost_reg_network(output_layer, y, theta1, theta2, M, reg):
    H = cost_netword(output_layer, y, M)

    return H + reg / (2 * M) * (np.sum(theta1[:, 1:] * theta1[:, 1:]) + np.sum(theta2[:, 1:] * theta2[:, 1:]))

def normalize(X):
    Mu = np.mean(X, axis=0)
    Sigma = np.std(X,axis=0)
    X_norm = (X - Mu) / Sigma

    return [X_norm, Mu, Sigma]

def normal_ecuation_method(X, Y):
    return np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def random_weights(L_in, L_out, E_ini=0.12):
    return np.random.uniform(-E_ini, E_ini, (L_in, L_out))

def forward_propagate(theta1, theta2, X):

    hidden_layer_z2 = np.matmul(X, np.transpose(theta1))
    hidden_layer_a2 = sigmoid(hidden_layer_z2)
    hidden_layer_a2 = np.hstack([np.ones([np.shape(hidden_layer_a2)[0], 1]), hidden_layer_a2])

    output_layer_z3 = np.matmul(hidden_layer_a2, np.transpose(theta2))
    output_layer_a3 = sigmoid(output_layer_z3)

    return X, hidden_layer_z2, hidden_layer_a2, output_layer_z3, output_layer_a3

def backprop(params_rn, num_entries, num_hidden, num_labels, X, y, reg):
    theta1 = np.reshape(params_rn[:num_hidden * (num_entries + 1)], (num_hidden, (num_entries + 1)))
    theta2 = np.reshape(params_rn[num_hidden * (num_entries + 1):], (num_labels, (num_hidden + 1)))
    M = len(y)

    X = np.hstack([np.ones([len(X), 1]), X])

    a1, z2, a2, z3, h = forward_propagate(theta1, theta2, X)

    cost = cost_reg_network(np.transpose(h), y, theta1, theta2, M, reg)

    m = X.shape[0]
    delta1, delta2 = np.zeros(np.shape(theta1)), np.zeros(np.shape(theta2))

    for t in range(m):
        a1t = a1[t, :]
        a2t = a2[t, :]
        ht = h[t,:]
        yt = y[t]
        d3t = ht-yt
        d2t = np.dot(theta2.T, d3t) * (a2t * (1 - a2t))
        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    gradient1 = delta1 / m
    gradient2 = delta2 / m

    reg1 = (reg / m) * theta1
    reg2 = (reg / m) * theta2

    reg1[:, 0] = 0
    reg2[:, 0] = 0

    gradient1 += reg1
    gradient2 += reg2

    gradient = np.concatenate((np.ravel(gradient1), np.ravel(gradient2)))

    return cost, gradient

def correct_classified(X, y, params_rn, num_entries, num_hidden, num_labels):
    X = np.hstack([np.ones([len(X), 1]), X])
    theta1 = np.reshape(params_rn[:num_hidden * (num_entries + 1)], (num_hidden, (num_entries + 1)))
    theta2 = np.reshape(params_rn[num_hidden * (num_entries + 1):], (num_labels, (num_hidden + 1)))

    hidden_layer = sigmoid(np.matmul(theta1, np.transpose(X)))
    hidden_layer = np.vstack([np.ones([len(X)]), hidden_layer])

    output_layer = sigmoid(np.matmul(theta2, hidden_layer))

    max_values = np.argmax(output_layer, axis=0)

    correct_values = y == max_values

    print("Se han clasificado correctamente: {} %".format(np.sum(correct_values) / len(correct_values) * 100))

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
   
    X_test, mu, sigma = normalize(X_test)
    input_layer_size = 20
    hidden_layer_size = 6
    output_layer_size = 1

    theta1 = random_weights(hidden_layer_size-1, input_layer_size)
    theta2 = random_weights(output_layer_size,hidden_layer_size)

    params_rn = np.concatenate((theta1.flatten(), theta2.flatten()))

    Y_test = np.reshape(Y_test,(len(Y_test),1))

    fmin = minimize(fun=backprop, x0=params_rn, args=(input_layer_size -1, hidden_layer_size-1, 1, X_test, Y_test, 1), method='TNC', jac=True, options={'maxiter': 70})
   
    print('El coste es de: {}'.format(fmin['fun']))

    correct_classified(X_test, Y_test, fmin.x, input_layer_size -1, hidden_layer_size-1, 1)

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

    test_size = int(0.4 * total_examples)
    val_size= total_examples-test_size

    X_test = X[0:test_size, :]
    X_val = X[test_size+1:total_examples, :]

    Y_test = Y[0:test_size] 
    Y_val = Y[test_size+1:total_examples]

    return X_test, X_val, Y_test, Y_val

def main():
    

    Data = load_csv('Práctica Final/Recursos/high_diamond_ranked.csv')

    X_test, X_val, Y_test, Y_val = split_data(Data)

    
    #lineal_regresion(X_test,Y_test, X_val, Y_val)
    #logistic_regresion(X_test,Y_test, X_val, Y_val)
    neural_network(X_test,Y_test, X_val, Y_val)
    #SVM(X_test,Y_test, X_val, Y_val)




main()