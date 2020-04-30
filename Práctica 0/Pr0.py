import numpy as np
import time
import matplotlib.pyplot as plt

def hyper_like_func(x):
    return (np.e ** x - np.e ** (-x)) / (x ** 3)

def integra_mc_eff(fun, a, b, num_puntos=600):
    tic = time.process_time()
    x = np.linspace(a, b, num_puntos)
    y = fun(x)

    plt.plot(x, y, c='blue')

    x_random = np.random.uniform(low=a, high=b, size=int(num_puntos))
    y_random = np.random.uniform(low=0, high=y.max(), size=int(num_puntos))

    plt.plot(x_random, y_random, 'x', c='red')

    print("The area is: {} %".format((np.sum(y_random < fun(x_random)) / num_puntos) * 100))

    toc = time.process_time()

    plt.savefig('Práctica 0/Recursos/solution.png')

    return 1000 * (toc - tic)

def integra_mc_in(fun, a, b, num_puntos=1000):
    tic = time.process_time()
    x = np.linspace(a, b, num_puntos)
    y = []

    for i in x:
        y += [fun(i)]

    x_random = np.random.uniform(low=a, high=b + 1, size=int(num_puntos))
    y_random = np.random.uniform(low=0, high=max(y) + 1, size=int(num_puntos))

    under_graph_points = 0

    for i in range(int(num_puntos)):
        if y_random[i] < fun(x_random[i]):
            under_graph_points += 1

    print("The area is: {} %".format((under_graph_points / num_puntos) * 100))

    toc = time.process_time()

    return 1000 * (toc - tic)

def main():
    fun, a, b = hyper_like_func, 1, 7

    points = np.linspace(100, 1000000, 20)

    y_efficient = []
    y_inefficient = []

    for i in points:
        y_efficient += [integra_mc_eff(fun, a, b, num_puntos=i)]
        y_inefficient += [integra_mc_in(fun, a, b, num_puntos=i)]

    plt.figure()
    plt.scatter(points, y_inefficient, c='red', label='inefficient')
    plt.scatter(points, y_efficient, c='blue', label='efficient')
    plt.legend()
    plt.savefig('time_graph.png')

    print("Done!")

fun, a, b = hyper_like_func, 1, 7

integra_mc_eff(fun, a, b)

#main()
