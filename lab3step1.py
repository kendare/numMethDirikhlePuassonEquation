import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tabulate import tabulate
import math


def setMatrix2(n, m):
    a1 = -1
    b1 = 1
    c1 = -1
    d1 = 1
    h = (b1-a1) / n
    k = (d1-c1)/ m
    a = 1 / h ** 2
    b = 1 / k ** 2
    A = -2 * (a + b)

    matrix = []
    for i in range((m - 1) * (n - 1)):
        matrix.append([0] * (m - 1) * (n - 1))

    for i in range((n - 1) * (m - 1)):
        matrix[i][i] = A
    for i in range((n - 1) * (m - 2)):
        matrix[i][i + n - 1] = b
    for i in range((n - 1) * (m - 2)):
        matrix[i + n - 1][i] = b
    for i in range((n - 1) * (m - 1) - 1):
        matrix[i + 1][i] = a
    for i in range((n - 1) * (m - 1) - 1):
        matrix[i][i + 1] = a

    for i in range(1, (m - 1)):
        matrix[i * (n - 1) - 1][i * (n - 1)] = 0

    for i in range(1, (m - 1)):
        matrix[i * (n - 1)][i * (n - 1) - 1] = 0

    return matrix



def set_solve_vector3(n, m):
    a1 = -1
    b1 = 1
    c1 = -1
    d1 = 1
    h = (b1-a1) / n
    k = (d1-c1)/ m

    m1_list = []
    m2_list = []
    m3_list = []
    m4_list = []
    f_list = []

    y = c1 + k
    x = a1 + h
    for i in range(m - 1):
        m1_list.append(-y**2+1)
        m2_list.append(-y**2+1)
        y += k

    for i in range(n - 1):
        m3_list.append(abs(np.sin(np.pi*x)))
        m4_list.append(abs(np.sin(np.pi*x)))
        x += h

    y = c1 + k
    x = a1 + h
    for i in range(m - 1):
        x += h
        y = c1
        for j in range(n - 1):
            y += k
            f_list.append(abs(np.sin(np.pi*x*y)**3))

    vector = []
    for i in range((n - 1) * (m - 1)):
        vector.append(-f_list[i])

    for i in range(len(m3_list)):
        vector[i] -= m3_list[i] / k ** 2
        vector[-1 - i] -= m4_list[-1 - i] / k ** 2

    for i in range(len(m1_list)):
        vector[i * len(m3_list)] -= m1_list[i] / h ** 2
        vector[i * len(m3_list) + len(m3_list) - 1] -= m2_list[i] / h ** 2

    return vector


def linear_interpolation(solve_vector, h):
    x = -1 + h
    interp = []
    for j in range(int(np.sqrt(len(solve_vector)))):
        for i in range(int(np.sqrt(len(solve_vector)))):
            interp.append(-x * (x + 1))
            x += h
        x = -1 + h
    return interp


def Zeydel_solve_test(n, m, N_max, eps, omega):
    # N_max = 10000
    S = 0
    # eps = 1e-12

    a = -1
    b = 1
    c = -1
    d = 1
    v_new = 0
    exit = False
    h2 = -(n / (b - a)) ** 2
    k2 = -(m / (d - c)) ** 2
    a2 = -2 * (h2 + k2)

    f_list = []
    f_test_list = []
    v_list = []
    r_list = []

    h = (b-a) / n
    k = (d-c) / m

    for i in range((m + 1)):
        f_list.append([0] * (n + 1))
        v_list.append([0] * (n + 1))
        r_list.append([0] * (n + 1))
        f_test_list.append([0] * (n + 1))

    # v_list.reverse()
    # f_list.reverse()

    y = c
    for i in range(m + 1):
        x = a
        for j in range(n + 1):
            f_list[i][j] = 4*(np.exp(1-x **2 -  y**2) *(x**2 + y**2 -1))
            v_list[i][j] = -np.exp(1-x **2 -  y**2)
            r_list[i][j] = -np.exp(1-x **2 -  y**2)
            f_test_list[i][j] = -np.exp(1-x **2 -  y**2)
            x += h
        y += k

    for i in range(1, m):
        for j in range(1, n):
            v_list[i][j] = 0
    # f_list.reverse()
    # print('f list:', f_list)
    # print('v list:', v_list)
    while exit != True:
        eps_max = 0
        for j in range(1, n):
            for i in range(1, m):
                v_old = v_list[i][j]
                v_new = -omega * (
                        h2 * (v_list[i + 1][j] + v_list[i - 1][j]) + k2 * (v_list[i][j + 1] + v_list[i][j - 1]))
                v_new = v_new + (1 - omega) * a2 * v_list[i][j] + omega * f_list[i][j]
                v_new /= a2
                eps_cur = abs(v_old - v_new)
                if eps_cur > eps_max:
                    eps_max = eps_cur
                v_list[i][j] = v_new
        S += 1
        
        if (eps_max < eps) or (S >= N_max):
            exit = True

    r = 0
    for i in range(m + 1):
        for j in range(n + 1):
            r_list[i][j] = abs(r_list[i][j] - v_list[i][j])


    matrix = np.array(r_list)
    max_index = np.argmax(r_list, axis=None)
    row_index, col_index = np.unravel_index(max_index, matrix.shape)

    Xmax = -1 + col_index * h
    Ymax = -1 + row_index * k


    return v_list, max(max(r_list)), S, eps_max, f_list, f_test_list, Xmax, Ymax, r_list
#1 v_list - Численное
#2 максимальное отклонение
#3 S - количество простых итераций
#4 eps_max - точность метода
#5 f_list - начальная функция
#6 f_test_list - аналитическое решение
#7 Xmax - максимальное значение по х
#8 Ymax - максимальное значение по у
#9 r_list - таблица разности численного и аналитического

def Zeydel_solve(n, m, N_max, eps, omega):
    # N_max = 10000
    S = 0
    # eps = 1e-12
    a = -1
    b = 1
    c = -1
    d = 1
    v_new = 0
    exit = False
    h2 = -(n / (b - a)) ** 2
    k2 = -(m / (d - c)) ** 2
    a2 = -2 * (h2 + k2)

    f_list = []
    v_list = []
    r_list = []

    h = (b-a) / n
    k = (d-c) / m

    for i in range((m + 1)):
        f_list.append([0] * (n + 1))
        v_list.append([0] * (n + 1))
        r_list.append([0] * (n + 1))

    # v_list.reverse()
    # f_list.reverse()

    # y = c + k
    # for i in range(1, m):
    #    x = -1 + h
    #    for j in range(1, n):
    #        f_list[i][j] = -np.cosh(x ** 2 * y)
    #        x += h
    #    y += k

    y = c
    for i in range(m + 1):
        x = a
        for j in range(n + 1):
            f_list[i][j] = abs(np.sin(np.pi*x*y)**3)
            x += h
        y += k
    r_list = f_list
    y = c
    for i in range(n + 1):
        v_list[0][i] = -y**2+1
        v_list[-1][i] = -y**2+1
        y += k

    x = a
    for i in range(m + 1):
        v_list[i][0] = abs(np.sin(np.pi*x))
        v_list[i][-1] = abs(np.sin(np.pi*x))
        x += h

    # f_list.reverse()
    # print('f list:', f_list)
    # print('v list:', v_list)
    while exit != True:
        eps_max = 0
        for j in range(1, n):
            for i in range(1, m):
                v_old = v_list[i][j]
                v_new = -omega * (
                        h2 * (v_list[i + 1][j] + v_list[i - 1][j]) + k2 * (v_list[i][j + 1] + v_list[i][j - 1]))
                v_new = v_new + (1 - omega) * a2 * v_list[i][j] + omega * f_list[i][j]
                v_new /= a2
                eps_cur = abs(v_old - v_new)
                if eps_cur > eps_max:
                    eps_max = eps_cur
                v_list[i][j] = v_new
        S += 1
        if (eps_max < eps) or (S >= N_max):
            exit = True
    for i in range(m + 1):
        for j in range(n + 1):
            r_list[i][j] = abs(r_list[i][j] - v_list[i][j])
    matrix = np.array(r_list)
    max_index = np.argmax(r_list, axis=None)
    row_index, col_index = np.unravel_index(max_index, matrix.shape)

    Xmax = -1 + col_index * h
    Ymax = -1 + row_index * k

    return v_list, r_list, S, eps_max
###
#1 v_list - численное решение основной задачи
#2 r_list - невязка СЛАУ
#3 S - количество итераций на решение СЛАУ
#3 EPS(N) - точность метода
#4 Xmax - максимальное значение по координате х
#4 Ymax - максимальное значение по координате у 
###


def splitMatrix(n, m):
    matrix = setMatrix2(n, m)
    MatrixL = []
    MatrixR = []
    MatrixD = []
    MatrixD1 = []
    for i in range((m - 1) * (n - 1)):
        MatrixL.append([0] * (m - 1) * (n - 1))
        MatrixR.append([0] * (m - 1) * (n - 1))
        MatrixD.append([0] * (m - 1) * (n - 1))
        MatrixD1.append([0] * (m - 1) * (n - 1))

    for i in range((m - 1) * (n - 1)):
        for j in range((m - 1) * (n - 1)):
            if i < j:
                MatrixR[i][j] = matrix[i][j]
            if i > j:
                MatrixL[i][j] = matrix[i][j]
        MatrixD[i][i] = matrix[i][i]
        MatrixD1[i][i] = 1 / matrix[i][i]
    return MatrixD, MatrixL, MatrixR, MatrixD1


def getB(n, m):
    D, L, R, D1 = splitMatrix(n, m)

    tmp = []
    for i in range((m - 1) * (n - 1)):
        tmp.append([0] * (m - 1) * (n - 1))
    for i in range((m - 1) * (n - 1)):
        for j in range((m - 1) * (n - 1)):
            tmp[i][j] = L[i][j] - R[i][j]
    x = np.dot(D1, tmp)
    return np.linalg.eig(x)[0]


def getOmega(n, m):
    v = getB(n, m)
    for i in range((n - 1) * (m - 1)):
        v[i] = abs(v[i])
    return 2 / (1 + np.sqrt(1 - (max(v) ** 2).real))






def getCoordinates(n, m, v):
    h = 2 / n
    k = 2 / m

    X = []
    Y = []
    Z = []
    for i in range(len(v)):
        for j in range(len(v[i])):
            X.append(-1 + i * h)
            Y.append(-1 + j * k)
            Z.append(v[i][j])
    return X, Y, Z


def plotGraph(X,Y,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

n = 20
m = 20

numOmega = getOmega(n,m)
#numOmega = 1.94898


v, r, S, eps, f_list, f, Xmax, Ymax, r_list = Zeydel_solve_test(n, m, 1000, 1e-12, numOmega)

#1 v_list - Численное
#2 максимальное отклонение
#3 S - количество простых итераций
#4 eps_max - точность метода
#5 f_list - начальная функция
#6 f_test_list - аналитическое решение
#7 Xmax - максимальное значение по х
#8 Ymax - максимальное значение по у
#9 r_list - таблица разности численного и аналитического
#####################################
###
#1 v_list - численное решение основной задачи
#2 r_list - невязка СЛАУ
#3 S - количество итераций на решение СЛАУ
#3 EPS(N) - точность метода
###

v2, r2_list, S2, EPS2 = Zeydel_solve(n, m, 1000, 1e-12, numOmega)
v3, r3_list, S3, EPS3 = Zeydel_solve(2 * n, 2 * m, 1000, 1e-12, numOmega)

raz_list = []
for i in range((m + 1)):
    raz_list.append([0] * (n + 1))
r1 = 0
for i in range(m + 1):
    for j in range(n + 1):
        raz_list[i][j] = abs(v2[i][j] - v3[(2 * i)][(2 * j)])
        if r1 < raz_list[i][j]:
            r1 = raz_list[i][j]
            ii = i
            jj = j


#max_index = np.argmax(raz_list, axis=None)
#row_index, col_index = np.unravel_index(max_index, matrix.shape)
#X1max = -1 + row_index * (2/n)
#Y1max = -1 + col_index * (2/m)
razMax = max(max(raz_list))
Y11max = -1 + jj * (2/m)
X11max = -1 + ii * (2/n)
#print(v)

plotGraph(*getCoordinates(n, m, v))
# численное решение тест
plotGraph(*getCoordinates(n, m, f))
# точное решение тест
plotGraph(*getCoordinates(n, m, r_list))
#  разность точного и численного решения тест

plotGraph(*getCoordinates(n, m, v2))
#График разности Vч
plotGraph(*getCoordinates(n, m, v3))
#График V2ч
plotGraph(*getCoordinates(n, m, raz_list))
#График разности Vч - V2ч


np.savetxt('mainR.csv', raz_list, fmt='%.8f')
np.savetxt('mainV.csv', v2, fmt='%.8f')
np.savetxt('mainV2.csv', v3, fmt='%.8f')
np.savetxt('testR.csv', r_list, fmt='%.8f')
np.savetxt('testV.csv', v, fmt='%.8f')
np.savetxt('testU.csv', f, fmt='%.8f') 

message = f"omega = {numOmega}\n"

message += f"Решение слау в тестовой задаче было найдено за {S} итераций\n"
message += f"EPS1 =  {r}\n"
message += f"Достигнута точность метода EPS(S) = {eps}\n"
message += f"Максимальное отклонение достигнуто в точке x = {Xmax} y = {Ymax}\n\n\n"

message += f"Решение слау в основной задаче задаче было найдено за {S2} итераций \n"
message += f"Достигнута точность метода EPS(S) = {EPS2}\n\n\n"

message += f"Решение слау в основной задаче с сеткой 2n задаче было найдено за {S3} итераций \n"
message += f"Достигнута точность метода EPS(S) = {EPS3}\n\n"
message += f"Максимальное отклонение численного решения на основной сетке и \nсетке с половинным шагом достигнуто в узле x = {X11max} y = {Y11max}\n"
message += f"и равно {r1}\n\n\n"

print(message)
'''
n = 240
m = 240
matrix = setMatrix2(n, m)
matrix = np.linalg.inv(matrix)
r1 = 0
for i in range(m + 1):
    for j in range(n + 1):
        matrix[i][j]
        if r1 < abs(matrix[i][j]):
            r1 = abs(matrix[i][j])

print(r1)


n = 120
m = 120
numOmega = 1.94898
matrix = setMatrix2(n, m)
solve = set_solve_vector3(n, m)
#V = Zeydel_solve(n, m, 1000, 1e-12, numOmega)[0]
#VTEST = Zeydel_solve_test(n, m, 1000, 1e-12, numOmega)[0]
V2 = Zeydel_solve(2*n, 2*m, 1000, 1e-12, numOmega)[0]
v3 = np.linalg.solve(matrix,solve)

def nevyazka(matrix, solve, v):
    v2 = []
    v3 = np.linalg.solve(matrix, solve)
    for i in range(1, len(v[0]) - 1):
        for j in range(1, len(v) - 1):
            v2.append(v[j][i])
    dot = np.dot(matrix, v2)
    nev = []
    for i in range(len(solve)):
        nev.append(abs(solve[i] - dot[i]))
    return nev

matrix2 = setMatrix2(2*n, 2*m)
solve2 = set_solve_vector3(2*n, 2*m)
#print(max(nevyazka(matrix, solve, VTEST)))
#print(max(nevyazka(matrix, solve, V)))
print(max(nevyazka(matrix2, solve2, V2)))
'''