import numpy
import matplotlib.pyplot as plt
import pandas as pd

# устанавливает размерность матрицы Q и вектора z.
N = 6
# устанавливает значение элемента a матрицы Q
a = 4
# устанавливает значения параметра gamma, для которых нужно решить систему уравнений.
gamma_values = [1.0, 0.5, 0.25, 0.125]

plot_data = {"gamma": [], "cond": [], "error": []}

# начинает цикл, в котором gamma перебирает значения из списка gamma_values
for gamma in gamma_values:
    # устанавливает значения переменных x и y в соответствии с текущим значением gamma
    x = 4 + gamma
    y = 4 - gamma

    # создает список g, содержащий значения г в соответствии с заданными формулами.
    g = [2 ** (k - 4) for k in range(1, N + 1)]

    # создает матрицу Q и вектор z, заполненные нулями.
    Q = numpy.zeros((N, N))
    z = numpy.zeros(N)

    # заполняет матрицу Q соответствующими значениями из заданных формул.
    for i in range(N):
        z[i] = y * sum(g[:i]) + a * g[i] + x * sum(g[i + 1:])
        for j in range(N):
            if i > j:
                Q[i][j] = y
            elif i == j:
                Q[i][j] = a
            else:
                Q[i][j] = x
    # вычисляет числа обусловленности матрицы Q.
    cond = numpy.linalg.cond(Q)
    P, R = numpy.linalg.qr(Q)
    #  использует для решения системы уравнений Qw = z с помощью метода LU-разложения
    w = numpy.linalg.solve(R, numpy.dot(P.T, z))
    # создает массив g, содержащий точные значения решения.
    g = numpy.array(g)
    # использует для вычисления относительной погрешности решения
    error = numpy.linalg.norm(w - g) / numpy.linalg.norm(g)
    # добавляет данные о текущем значении gamma в соответствующие списки
    plot_data["gamma"].append(gamma)
    plot_data["cond"].append(cond)
    plot_data["error"].append(error)

# Построение таблицы
results = pd.DataFrame({
    'Gamma': plot_data["gamma"],
    'cond(Q)': plot_data["cond"],
    'Error': plot_data["error"]
})

print(results.to_string(index=False))

# Построение графика
plt.plot(plot_data["gamma"], plot_data["cond"], label="cond(Q)")
plt.plot(plot_data["gamma"], plot_data["error"], label="Error")
plt.xlabel("Gamma")
plt.legend()
plt.savefig("img.jpg")
plt.show()
