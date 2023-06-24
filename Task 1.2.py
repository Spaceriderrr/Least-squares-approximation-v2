import numpy as np
import scipy
import matplotlib.pyplot as plt


def func(x, a, b):
    return a * x / (b + x)


def func_remade(x, a, b):
    return 1 / a + b / (x * a)


with open('x4.txt', 'rt', encoding='utf-8') as x_file, open('y4.txt', 'rt', encoding='utf-8') as y_file:
    x_points = list(map(float, x_file.read().split()))
    y_points = list(map(float, y_file.read().split()))

t_points = [1 / y for y in y_points]

A = np.vstack((x_points, np.ones(len(x_points)))).T
a, b = scipy.linalg.lstsq(A, np.array(y_points))[0]

B = np.vstack((x_points[1::], np.ones(len(x_points) - 1))).T
m, n = scipy.linalg.lstsq(B, np.array(t_points[1::]))[0]

alpha, beta = scipy.optimize.curve_fit(func, xdata=x_points[1::], ydata=y_points[1::])[0]

fig, ax = plt.subplots()

ax.set_xlim(-0.25, 5.25)
ax.set_ylim(-0.25, 2.5)

plt.scatter(x_points, y_points, s=2, c='red', label='data')

plt.plot(x_points, [func(x, a, b) for x in x_points], c='green', label=f'a={round(a, 2)},b={round(b, 2)}, scipy, lstsq')
plt.plot(x_points[1::], [func_remade(x, m, n) for x in x_points[1::]], c='magenta', label=f'a={round(m, 2)}'
                                                                                          f',b={round(n, 2)},'
                                                                                          f' scipy, lstsq '
                                                                                          f'(transformed)')
plt.plot(x_points[1::], [func(x, alpha, beta) for x in x_points[1::]], c='blue', label=f'a={round(alpha, 2)},'
                                                                                       f'b={round(beta, 2)}, scipy, '
                                                                                       f'curve_fit')
plt.legend(loc='center right')
plt.grid(True)
plt.show()
