import numpy as np
import matplotlib.pyplot as plt
import math


def hessian_matrix(f, x, deltaX):
    fx = f(x)
    N = len(x)
    H = []
    for i in range(N):
        hi = []
        for j in range(N):
            if i == j:
                xp = x.copy()
                xn = x.copy()
                xp[i] = xp[i] + deltaX
                xn[i] = xn[i] - deltaX
                hi.append((f(xp) - 2 * fx + f(xn)) / (deltaX ** 2))
            else:
                xpp = x.copy()
                xpn = x.copy()
                xnp = x.copy()
                xnn = x.copy()
                xpp[i] = xpp[i] + deltaX
                xpp[j] = xpp[j] + deltaX
                xpn[i] = xpn[i] + deltaX
                xpn[j] = xpn[j] - deltaX
                xnp[i] = xnp[i] - deltaX
                xnp[j] = xnp[j] + deltaX
                xnn[i] = xnn[i] - deltaX
                xnn[j] = xnn[j] - deltaX
                hi.append((f(xpp) - f(xpn) - f(xnp) + f(xnn)) / (4 * deltaX ** 2))
        H.append(hi)
    return np.array(H)

def regla_eliminacion(x1, x2, fx1, fx2, a, b):
    if fx1 > fx2:
        return x1, b
    if fx1 < fx2:
        return a, x2
    return x1, x2 

def w_to_x(w, a, b):
    return w * (b - a) + a 

def busquedaDorada(funcion, epsilon, a=None, b=None):
    phi = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1
    while Lw > epsilon:
        w2 = aw + phi * Lw
        w1 = bw - phi * Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), funcion(w_to_x(w2, a, b)), aw, bw)
        k += 1
        Lw = bw - aw
    return (w_to_x(aw, a, b) + w_to_x(bw, a, b)) / 2

def gradiente(f, x, deltaX=0.001):
    grad = []
    for i in range(len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i] + deltaX
        xn[i] = xn[i] - deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return np.array(grad)
def newton_with_trajectory(f, x0, epsilon1, epsilon2, M):
    terminar = False
    xk = x0
    k = 0
    trajectory = [x0]

    while not terminar:
        grad = np.array(gradiente(f, xk))
        hessian = hessian_matrix(f, xk, deltaX=0.001)
        hessian_inv = np.linalg.inv(hessian)

        if np.linalg.norm(grad) < epsilon1 or k >= M:
            terminar = True
        else:
            def alpha_funcion(alpha):
                return f(xk - alpha * np.dot(hessian_inv, grad))

            alpha = busquedaDorada(alpha_funcion, epsilon=epsilon2, a=0.0, b=1.0)
            x_k1 = xk - alpha * np.dot(hessian_inv, grad)

            if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 0.00001) <= epsilon2:
                terminar = True
            else:
                k += 1
                xk = x_k1
                trajectory.append(x_k1)
    return xk, np.array(trajectory)

def minimo_maximo(f, x, deltaX=0.001):
    H = hessian_matrix(f, x, deltaX)
    eigenvalues = np.linalg.eigvals(H)
    if all(eigenvalues > 0):
        return "Minimo"
    elif all(eigenvalues < 0):
        return "Maximo"
    else:
        return "punto de silla"


# x0 = np.array([3.0, 2.0])
x0 = np.array([0.0, 0.0])
e1 = 0.001
e2 = 0.001
max_iter = 100

himmenblau = lambda x: (((x[0] ** 2) + x[1] - 11) ** 2) + ((x[0] + (x[1] ** 2) - 7) ** 2)
resultado, trajectory = newton_with_trajectory(himmenblau, x0, e1, e2, max_iter)
print("Resultado:", resultado)
print("Tipo de punto crítico:", minimo_maximo(himmenblau, resultado))

# Graficar la función y la trayectoria del punto óptimo
x = np.linspace(-6, 6, 100)
y = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x, y)
Z = himmenblau([X, Y])

plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis', alpha=0.7)
plt.plot(trajectory[:,0], trajectory[:,1], marker='o', color='r', label='Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Newton-Raphson Optimization Trajectory')
plt.legend()
plt.colorbar(label='log(f(x, y))')
plt.show()
