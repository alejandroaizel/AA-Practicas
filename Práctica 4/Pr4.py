import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
from scipy.optimize import minimize


def debugInitializeWeights(fan_in, fan_out):
    """
    Initializes the weights of a layer with fan_in incoming connections and
    fan_out outgoing connections using a fixed set of values.
    """

    # Set W to zero matrix
    W = np.zeros((fan_out, fan_in + 1))

    # Initialize W using "sin". This ensures that W is always of the same
    # values and will be useful in debugging.
    W = np.array([np.sin(w) for w in
                  range(np.size(W))]).reshape((np.size(W, 0), np.size(W, 1)))

    return W

def computeNumericalGradient(J, theta):
    """
    Computes the gradient of J around theta using finite differences and
    yields a numerical estimate of the gradient.
    """

    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4

    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * tol)
        perturb[p] = 0

    return numgrad


def checkNNGradients(costNN, reg_param):
    """
    Creates a small neural network to check the back propogation gradients.
    Outputs the analytical gradients produced by the back prop code and the
    numerical gradients computed using the computeNumericalGradient function.
    These should result in very similar values.
    """
    # Set up small NN
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate some random test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to get random X
    X = debugInitializeWeights(input_layer_size - 1, m)

    # Set each element of y to be in [0,num_labels]
    y = [(i % num_labels) for i in range(m)]

    ys = np.zeros((m, num_labels))
    for i in range(m):
        ys[i, y[i]] = 1

    # Unroll parameters
    nn_params = np.append(Theta1, Theta2).reshape(-1)

    # Compute Cost
    cost, grad = costNN(nn_params,
                        input_layer_size,
                        hidden_layer_size,
                        num_labels,
                        X, ys, reg_param)

    def reduced_cost_func(p):
        """ Cheaply decorated nnCostFunction """
        return costNN(p, input_layer_size, hidden_layer_size, num_labels,
                      X, ys, reg_param)[0]

    numgrad = computeNumericalGradient(reduced_cost_func, nn_params)

    # Check two gradients
    np.testing.assert_almost_equal(grad, numgrad)
    
    return (grad - numgrad)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_der(X):
    return sigmoid(X) * (1 - sigmoid(X))

def cost(output_layer, y, M):
    return 1 / M * np.trace((np.dot(-y, np.log(output_layer)) - np.dot((1 - y), np.log(1 - output_layer))))

def cost_reg(output_layer, y, theta1, theta2, M, reg):
    H = cost(output_layer, y, M)

    return H + reg / (2 * M) * (np.sum(theta1[:, 1:] * theta1[:, 1:]) + np.sum(theta2[:, 1:] * theta2[:, 1:]))

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

    cost = cost_reg(np.transpose(h), y, theta1, theta2, M, reg)

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

def main():
    data = loadmat('Práctica 4/Recursos/ex4data1.mat')
    weights = loadmat('Práctica 4/Recursos/ex4weights.mat')

    theta1_sol, theta2_sol = weights['Theta1'], weights['Theta2']

    X = data['X']
    y = data['y'].ravel()

    m = len(y)
    input_size = X.shape[1]
    num_hidden = 25
    num_labels = 10
    reg = 1
    
    y = (y - 1)
    y_onehot = np.zeros((m, num_labels))

    for i in range(m):
        y_onehot[i][y[i]] = 1

    theta1 = random_weights(len(theta1_sol), len(theta1_sol[0]))
    theta2 = random_weights(len(theta2_sol), len(theta2_sol[0]))
    
    params_rn = np.concatenate((theta1.flatten(), theta2.flatten()))

    difference = checkNNGradients(backprop, reg)

    print('La mayor diferencia es de: {}'.format(max(difference)))
    print('La menor diferencia es de: {}'.format(min(difference)))

    fmin = minimize(fun=backprop, x0=params_rn, args=(input_size, num_hidden, num_labels, X, y_onehot, reg), method='TNC', jac=True, options={'maxiter': 70})

    print('El coste es de: {}'.format(fmin['fun']))

    correct_classified(X, y, fmin.x, input_size, num_hidden, num_labels)


main()