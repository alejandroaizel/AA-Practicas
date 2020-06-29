import numpy as np
import matplotlib.pyplot as plt
import scipy
import Recursos.process_email as pe
import Recursos.get_vocab_dict as gvd
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm as sk
import codecs as cod

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
    plt.savefig('Práctica 6/Recursos/{}.png'.format(name), dpi=200)
    plt.close()

# 1.1. Kernel lineal
def lineal_kernel():
    data = loadmat('Práctica 6/Recursos/ex6data1.mat')

    X, Y = data['X'], data['y'].ravel()

    svm = sk.SVC(kernel='linear', C=1)
    svm.fit(X,Y)

    visualize_boundary(X, Y, svm,'lineal_kernel_c_1')

    svm = sk.SVC(kernel='linear', C=100)
    svm.fit(X, Y)

    visualize_boundary(X, Y, svm, 'lineal_kernel_c_100')

# 1.2. Kernel gaussiano
def gaussian_kernel():
    data = loadmat('Práctica 6/Recursos/ex6data2.mat')

    X, Y = data['X'], data['y'].ravel()

    sigma = 0.1

    svm = sk.SVC(kernel='rbf', C=1, gamma=1 / (2 * sigma ** 2))
    svm.fit(X, Y)

    visualize_boundary(X, Y, svm, 'gaussian_kernel_c_1_s_0.1')

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

    visualize_boundary(X, Y, best, 'parameter_selection')


def make_data(directory, num_emails_begin, num_emails_end, vocab_dict):
    email_features = np.zeros((num_emails_end - num_emails_begin, len(vocab_dict)))

    for i in range(num_emails_begin, num_emails_end):
        email_contents = cod.open('Práctica 6/Recursos/{}/{}.txt'.format(directory, f'{(i + 1):04}'), 'r',
            encoding='utf-8', errors='ignore').read()
        email = pe.email2TokenList(email_contents)

        for w in email:
            if w in vocab_dict:
                email_features[i - num_emails_begin, vocab_dict[w] - 1] = 1

    return email_features

# 2. Detección de spam
def training_set(num_spam_train, num_easy_ham_train, num_hard_ham_train, name):
    vocab_dict = gvd.getVocabDict()

    X_train_spam = make_data("spam", 0, num_spam_train, vocab_dict)
    Y_train_spam = np.ones(num_spam_train)

    X_train_easy_ham = make_data("easy_ham", 0, num_easy_ham_train, vocab_dict)
    Y_train_easy_ham = np.zeros(num_easy_ham_train)

    X_train_hard_ham = make_data("hard_ham", 0, num_hard_ham_train, vocab_dict)
    Y_train_hard_ham = np.zeros(num_hard_ham_train)

    X_train = np.concatenate((X_train_spam, X_train_easy_ham, X_train_hard_ham), axis=0)
    Y_train = np.concatenate([Y_train_spam, Y_train_easy_ham, Y_train_hard_ham])

    X_val_spam = make_data("spam", num_spam_train, 500, vocab_dict)
    Y_val_spam = np.ones(500 - num_spam_train)

    X_val_easy_ham = make_data("easy_ham", num_easy_ham_train, 1551, vocab_dict)
    Y_val_easy_ham = np.zeros(1551 - num_easy_ham_train)

    X_val_hard_ham = make_data("hard_ham", num_hard_ham_train, 250, vocab_dict)
    Y_val_hard_ham = np.zeros(250 - num_hard_ham_train)

    X_val = np.concatenate((X_val_spam, X_val_easy_ham, X_val_hard_ham), axis=0)
    Y_val = np.concatenate([Y_val_spam, Y_val_easy_ham, Y_val_hard_ham])

    svm = sk.SVC(kernel='linear', C=1)
    svm.fit(X_train, Y_train)

    accuracy = svm.score(X_val, Y_val)

    print("\n{} training set:".format(name))
    print("   Correctly classified for C=1: {} %".format(round(accuracy * 100, 2)))

    svm = sk.SVC(kernel='linear', C=100)
    svm.fit(X_train, Y_train)

    accuracy = svm.score(X_val, Y_val)

    print("   Correctly classified for C=100: {} %".format(round(accuracy * 100, 2)))


def support_vector_machines():
    lineal_kernel()
    gaussian_kernel()
    parameter_selection()

def spam_detection():
    training_set(500, 1551, 0, "first")
    training_set(400, 1300, 150, "second")
    training_set(350, 1200, 250, "third")


def main():
    support_vector_machines()
    spam_detection()

main()