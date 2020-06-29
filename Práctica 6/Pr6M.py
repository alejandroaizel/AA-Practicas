import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import loadmat
from sklearn import svm
import re
import nltk
import nltk.stem.porter
import codecs as cod


def preProcess(email):
    
    hdrstart = email.find("\n\n")
    if hdrstart != -1:
        email = email[hdrstart:]

    email = email.lower()
    # Strip html tags. replace with a space
    email = re.sub('<[^<>]+>', ' ', email)
    # Any numbers get replaced with the string 'number'
    email = re.sub('[0-9]+', 'number', email)
    # Anything starting with http or https:// replaced with 'httpaddr'
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    # Strings with "@" in the middle are considered emails --> 'emailaddr'
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    # The '$' sign gets replaced with 'dollar'
    email = re.sub('[$]+', 'dollar', email)
    return email
def email2TokenList(raw_email):
    """
    Function that takes in a raw email, preprocesses it, tokenizes it,
    stems each word, and returns a list of tokens in the e-mail
    """

    stemmer = nltk.stem.porter.PorterStemmer()
    email = preProcess(raw_email)

    # Split the e-mail into individual words (tokens) 
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]',
                      email)

    # Loop over each token and use a stemmer to shorten it
    tokenlist = []
    for token in tokens:

        token = re.sub('[^a-zA-Z0-9]', '', token)
        stemmed = stemmer.stem(token)
        #Throw out empty tokens
        if not len(token):
            continue
        # Store a list of all unique stemmed words
        tokenlist.append(stemmed)

    return tokenlist
def getVocabDict(reverse=False):
    """
    Function to read in the supplied vocab list text file into a dictionary.
    Dictionary key is the stemmed word, value is the index in the text file
    If "reverse", the keys and values are switched.
    """
    vocab_dict = {}
    with open("Práctica 6/vocab.txt") as f:
        for line in f:
            (val, key) = line.split()
            if not reverse:
                vocab_dict[key] = int(val)
            else:
                vocab_dict[int(val)] = key

    return vocab_dict


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


def first_dataset():
    data = loadmat('Práctica 6/Recursos/ex6data1.mat')
    X, y = data['X'], data['y'].ravel()

    c = svm.SVC(kernel='linear',C=1)
    c.fit(X,y) 
    visualize_boundary(X,y,c,'visualize boundary with c 1.0')
    
    c = svm.SVC(kernel='linear',C=100)
    c.fit(X,y)
    visualize_boundary(X,y,c,'visualize boundary with c 100')


def second_dataset():

    data = loadmat('Práctica 6/Recursos/ex6data2.mat')
    X, y = data['X'], data['y'].ravel()

    C=1
    sigma=0.1
    c = svm.SVC(kernel='rbf', C=C,gamma=1 / (2 * sigma**2))
    c.fit(X,y)
    visualize_boundary(X,y,c,'visualize boundary with gaussian kernel')


def third_dataset():
    data = loadmat('Práctica 6/Recursos/ex6data3.mat')
    X, y = data['X'], data['y'].ravel()
    Xval, yval = data['Xval'], data['yval'].ravel()

    values =[ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    C_aux, sigma_aux = np.meshgrid(values, values)

    combinations = np.vstack((C_aux.ravel(), sigma_aux.ravel())).T # una matriz de (64,2) para guardar todas las combinaciones de C y sigma
    errors=[]
    for v in combinations:
        c = svm.SVC(kernel='rbf', C=v[0],gamma=1 / (2 * v[1]**2))
        c.fit(X,y)
        errors.append(c.score(Xval,yval)) # calcula el acierto medio entre el modelo calculado anteriormente y los conjuntos Xval, yval
    
    
    best_value = combinations[np.argmax(errors)]

    best = svm.SVC(kernel='rbf', C=best_value[0],gamma=1 / (2 * best_value[1]**2))
    best.fit(X,y)
    visualize_boundary(X,y,best,'Choosing parameters C and Sigma')
    
    print("Best values are :\n C : {} \n Sigma : {}".format(best_value[0],best_value[1]))


def make_data(route, start, end):
    '''
    Esta funcion se usa para devolver una matriz de 0's y 1's de dimension [(end-start) x 1899] en la que cada fila es un correo, y cada una de las 
    columnas es cada una de las palabras que aparecen en el correo.
    De esta forma generamos el conjunto X_train y X_val
    '''
    for i in range(start,end): 
        name = route + "0{}.txt".format(i) # los ceros me tocan la polla
        
        email_contents = cod.open(name , 'r' , encoding='utf8' , errors= 'ignore').read()
        email_words = email2TokenList(email_contents)

        dicc = getVocabDict() # no tocar

        emails_features = np.zeros(1900)
        feature_vector = [0] * (len(dicc)+1)# vector de 1899 posiciones(el +1 es por que len(..) es 1899 y la ultima posicion es 1898(la palabra zip lo jode), con el +1 llegamos a 1899)
        for word in email_words:
            if word in dicc:
                feature_vector[dicc[word]] = 1 # creo el vector de ceros y unos

        emails_features = np.vstack((emails_features,feature_vector)) # los añado a una matriz 


    return emails_features[1:,:] # para no devolver el primer vector de ceros

def spam_detector():

    # 500 correos spam
    # 250 hard ham
    # 2551 easy ham
    # Usaré la mitad de spam para entrenamiento y la segunda mitad para validacion, same para easy ham
    '''
    Lo que se me ocurrió es : vamos a tener una matriz de dimensiones enormes en la que esten consecutivos correos de spam y correos de ham, de forma
    que las 250 primeras filas seran correos de spam/ y las siguientes 250 filas seran correos de easy_ham/. Esto compone la matriz de entrenamiento.

    Los 250 correos restantes de spam/ más otros 250 correos de easy_ham/ compondran la matriz de validacion cruzada.

    El vector columna "y" que se usa para comprobar que todo vaya bien lo construimos nosotros, un 1 en el vector indica spam, un 0 ham. 
    Así,  las 250 priemras filas seran 1 y las siguientes 250 serán ceros. 

    Con esto podemos instanciar el SVM y hacer un fit(X_train,y) y para calcular el acierto score(X_val, y_val) 

    '''
    X_train = make_data("Práctica 6/Recursos/spam/", 100,250) # metemos 150 correos de spam/
    X_train.append( make_data("Práctica 6/Recursos/easy_ham/",100,250)) # metemos otros 250 correos de easy_ham # como esta puta mierda dvuelve un array me jode el .append
    y_train = [0] * 250
    y_train.append([1] * 250) 

    X_val = make_data("Práctica 6/Recursos/spam/",251,401)
    X_val.append(make_data("Práctica 6/Recursos/easy_ham/",251,400))
    y_val = [0] * 250
    y_val.append([1] * 250) 

    c = svm.SVC(kernel='linear',C=1)
    c.fit(X_train, y_train)

    accuracy = c.score(X_val, y_val)

    print("Correctly classifed : {}".format(accuracy))



def main():

    #first_dataset()
    #second_dataset()
    #third_dataset()
    
    spam_detector()

main()