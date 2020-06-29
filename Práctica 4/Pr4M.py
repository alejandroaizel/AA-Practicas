import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


def cost(theta1,theta2,X, h,onehot,landa):

   # reg = (landa/(2*len(X.shape[0]))) * (sum(sum((theta1[:,2])**2)) + sum(sum((theta2[:,2])**2)))
    C = ((1/X.shape[0])*sum(sum(-onehot * np.log(h) - ((1-onehot) * np.log(1-h)) )))# + reg
    return C

def sigmoid(X): 

    return 1 / (1 + np.exp(-X))

def forward_propagate(X,theta1,theta2):

   

    X = np.hstack([np.ones([len(X), 1]), X])  # columna de unos, X = [5000, 401]

    a1 = X

    z2 = np.dot(theta1, np.transpose(a1))
    a2_aux = sigmoid(z2) 

    add = np.array([np.ones([len(X)])])
    a2 = np.vstack((add, a2_aux))   

    z3 = np.dot(theta2, a2)

    h = sigmoid(z3)

    return a1, z2, a2,z3,h



def retro_propagacion(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y , reg):

    # a la X ya se le ha añadido una columna de unos
    # inicializacion de parametros

    theta1 = np.reshape(params_rn[:num_ocultas*(num_entradas+1)], (num_ocultas,(num_entradas+1)))
    theta2 = np.reshape(params_rn[:num_ocultas*(num_entradas+1)], (num_ocultas,(num_entradas+1)))
    gradiente_uno = np.zeros(theta1.shape)
    gradiente_dos = np.zeros(theta2.shape)

    m = X.shape(0) #5000

    for t in range(m):

        a1, z2, a2, z3, h = forward_propagate(X[1],theta1,theta2) # esto es lo de la practica 3
        # aqui se hace lo nuevo

        a1t = a1[t, :]
        a2t = a2[t, :]

        d3 = h-y
        d2_aux = np.dot(np.transpose(theta2),d3  ) * (a2t * (1 - a2t))
        d2 = d2_aux[:,1:] # eliminar la columna de unos

        delta1 = delta1+np.dot(d2[1:,np.newaxis],a1t[np.newaxis,:])
        delta2 = delta2 + np.dot(d3[:, np.newaxis], a2t[np.newaxis, :])

        #JCost = cost(theta1,theta2,X,h,onehot,landa) Esto da error pero la kbsa ya no me da pa mas
    
    
    thetaVec = np.concatenate((np.ravel(theta1)),(theta2))



   
def learning_network(X,y,theta1,theta2,landa):

    thetaVec = np.concatenate((np.ravel(theta1)),(theta2)) # desplegar las thetas en un vector

    fmin = minimize(retro_propagacion,thetaVec,args=(theta1.shape[1]-1,theta1.shape[0]+1,theta2.shape[0]), method='TCN',jac=True,options={'maxiter' : 70})



def pesos_aleatorios(L_in,L_out):

    limit = 0.12
    #no hace falta hacerle un +1 a L_in, ya llega hecho
    return np.random.uniform(low= -limit, high=limit, size=(L_out,L_in))

def main():

    weights = loadmat('ex4weights.mat')
    theta1_correcto, theta2_correcto = weights['Theta1'], weights['Theta2']
    #Theta1 es de dimension 25 x 401
    #Theta2 es de dimension 10 x 26
    datos = loadmat('ex4data1.mat')
    y = datos['y'].ravel()
    X = datos['X']
   
    m = len(y)
    for i in range(1,11):
        y[y==i]=i-1

    onehot = np.zeros((m,10))
    for i in range(m):
        onehot[i][y[i]]=1

    landa = 1
   
    #print(theta1_correcto.shape[1])
    theta1 = pesos_aleatorios(theta1_correcto.shape[1],theta1_correcto.shape[0])
    theta2 = pesos_aleatorios(theta1_correcto.shape[1],theta1_correcto.shape[0])

    learning_network(X,y,theta1,theta2,landa)



main()