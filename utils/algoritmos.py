"""Algoritmos básicos de cálculo numérico."""

import numpy as np


def bissecao(f, a, b, tol=1e-10, max_iter=100):
    """Encontra uma raiz de f em [a, b] pelo método da bisseção."""
    fa = f(a)
    fb = f(b)

    if fa == 0:
        return a, 0
    if fb == 0:
        return b, 0
    if fa * fb > 0:
        raise ValueError("O intervalo deve conter uma mudança de sinal.")

    for iteration in range(1, max_iter + 1):
        midpoint = (a + b) / 2.0
        fm = f(midpoint)

        if fm == 0 or (b - a) / 2 <= tol:
            return midpoint, iteration

        if fa * fm < 0:
            b, fb = midpoint, fm
        else:
            a, fa = midpoint, fm

    return (a + b) / 2.0, max_iter


def pontofixo(a, g, TOL=1e-8, max_iter=1000):
    """Encontra um ponto fixo de g a partir de a."""
    for _ in range(max_iter):
        x = g(a)
        if abs(x - a) <= TOL:
            return x
        a = x
    raise RuntimeError("O método do ponto fixo não convergiu.")


def newton_raphson(a, f, TOL=1e-8, df=None, max_iter=100):
    """Encontra uma raiz de f pelo método de Newton-Raphson."""
    for _ in range(max_iter):
        derivada = df(a) if df is not None else (
            f(a + TOL) - f(a - TOL)
        ) / (2 * TOL)
        if derivada == 0:
            raise ZeroDivisionError("A derivada se anulou durante a iteração.")
        x = a - f(a) / derivada
        if abs(x - a) <= TOL:
            return x
        a = x
    raise RuntimeError("O método de Newton-Raphson não convergiu.")


def jacobiano_numerico(F, x, eps=1e-8):
    """Aproxima numericamente o Jacobiano de um sistema F(x)=0 em x."""
    x = np.asarray(x, dtype=float)
    n = x.size
    J = np.zeros((n, n), dtype=float)

    for j in range(n):
        h = eps * max(1.0, abs(x[j]))
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h

        F_plus = np.asarray(F(x_plus), dtype=float)
        F_minus = np.asarray(F(x_minus), dtype=float)
        J[:, j] = (F_plus - F_minus) / (2.0 * h)

    return J


def newton_sistemas(x0, F, J=None, tol=1e-8, max_iter=100):
    """Resolve um sistema F(x)=0 pelo método de Newton-Raphson."""
    x = np.asarray(x0, dtype=float).copy()
    F_ = F
    J_ = J if J is not None else (lambda y: jacobiano_numerico(F_, y))

    for _ in range(max_iter):
        fx = np.asarray(F_(x), dtype=float)
        if fx.ndim != 1:
            raise ValueError("A função F deve retornar um vetor 1D.")
        if np.linalg.norm(fx, ord=np.inf) <= tol:
            return x

        jac = np.asarray(J_(x), dtype=float)
        if jac.shape != (x.size, x.size):
            raise ValueError("O Jacobiano deve ter dimensões (n, n).")

        delta = np.linalg.solve(jac, -fx)
        x = x + delta

        if np.linalg.norm(delta, ord=np.inf) <= tol:
            return x

    raise RuntimeError("O método de Newton-Raphson para sistemas não convergiu.")


def secante(a, b, f, TOL=1e-8, max_iter=100):
    """Encontra uma raiz de f pelo método da secante."""
    for _ in range(max_iter):
        fb = f(b)
        denominador = fb - f(a)
        if denominador == 0:
            raise ZeroDivisionError("O denominador da secante se anulou.")
        x = (a * fb - b * f(a)) / denominador
        if abs(x - b) <= TOL:
            return x
        a, b = b, x
    raise RuntimeError("O método da secante não convergiu.")


def lu_pivot(A):
    """Fatora A como PA = LU usando pivotamento parcial."""
    A = A.astype(float).copy()
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for k in range(n):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if np.isclose(U[pivot, k], 0.0):
            raise np.linalg.LinAlgError("Matriz singular ou quase singular.")
        if pivot != k:
            U[[k, pivot], k:] = U[[pivot, k], k:]
            P[[k, pivot], :] = P[[pivot, k], :]
            L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    np.fill_diagonal(L, 1.0)
    return P, L, U


def _substituicao_direta(L, B):
    Y = np.zeros(L.shape[0])
    for i in range(L.shape[0]):
        Y[i] = B[i] - np.dot(L[i, :i], Y[:i])
    return Y


def _substituicao_reversa(U, Y):
    X = np.zeros(U.shape[0])
    for i in reversed(range(U.shape[0])):
        if np.isclose(U[i, i], 0.0):
            raise np.linalg.LinAlgError("U possui pivô nulo.")
        X[i] = (Y[i] - np.dot(U[i, i + 1:], X[i + 1:])) / U[i, i]
    return X


def lu(A, B):
    """Resolve Ax = B por fatoração LU com pivotamento parcial."""
    P, L, U = lu_pivot(A)
    return _substituicao_reversa(U, _substituicao_direta(L, P @ B))


def jacobi(A, B, k, TOL):
    """Resolve Ax = B pelo método iterativo de Jacobi."""
    A = A.astype(float)
    B = B.astype(float)
    diagonal = np.diag(A)
    if np.any(np.isclose(diagonal, 0.0)):
        raise ValueError("A possui elementos diagonais nulos.")

    X = np.zeros(B.shape[0])
    resto = A - np.diagflat(diagonal)
    for _ in range(k):
        X_novo = (B - resto @ X) / diagonal
        if np.linalg.norm(X_novo - X, ord=2) < TOL:
            return X_novo
        X = X_novo
    return X


def seidel(A, B, k, TOL):
    """Resolve Ax = B pelo método iterativo de Gauss-Seidel."""
    A = A.astype(float)
    B = B.astype(float)
    if np.any(np.isclose(np.diag(A), 0.0)):
        raise ValueError("A possui elementos diagonais nulos.")

    X = np.zeros(B.shape[0])
    for _ in range(k):
        anterior = X.copy()
        for i in range(B.shape[0]):
            inferior = np.dot(A[i, :i], X[:i])
            superior = np.dot(A[i, i + 1:], X[i + 1:])
            X[i] = (B[i] - inferior - superior) / A[i, i]
        if np.linalg.norm(X - anterior, ord=2) < TOL:
            return X
    return X


def newton_coef(x, y):
    """Calcula os coeficientes divididos de Newton."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    a = y.copy()
    for j in range(1, n):
        a[j:n] = (a[j:n] - a[j - 1 : n - 1]) / (x[j:n] - x[: n - j])
    return a


def newton(x, y):
    """Retorna o polinômio interpolador de Newton em coeficientes."""
    a = newton_coef(x, y)
    p = np.array([a[-1]], dtype=float)
    for k in range(len(a) - 2, -1, -1):
        p = np.convolve(p, np.array([1.0, -x[k]], dtype=float))
        if len(p) > 0:
            p[-1] += a[k]
    return p


def lagrange_add(p, q):
    """Soma dois polinômios representados por coeficientes."""
    if len(q) > len(p):
        p, q = q, p
    off = len(p) - len(q)
    res = p[:]
    for i in range(len(q)):
        res[off + i] += q[i]
    return res


def lagrange_mul(p, q):
    """Multiplica dois polinômios representados por coeficientes."""
    deg = (len(p) - 1) + (len(q) - 1)
    res = [0.0] * (deg + 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            res[i + j] += a * b
    return res


def lagrange_scale(p, s):
    """Multiplica um polinômio por um escalar."""
    return [s * c for c in p]


def lagrange(x, y):
    """Retorna o polinômio interpolador de Lagrange em coeficientes."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    p = [0.0]
    for i in range(n):
        m = [1.0]
        d = 1.0
        xi = x[i]
        for j in range(n):
            if j == i:
                continue
            m = lagrange_mul(m, [1.0, -x[j]])
            d *= xi - x[j]
        pi = lagrange_scale(m, y[i] / d)
        p = lagrange_add(p, pi)
    return np.asarray(p, dtype=float)


def polinomial(x, y):
    """Retorna o polinômio interpolador via resolução do sistema Vandermonde."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    V = np.vander(x, N=n, increasing=False)
    return np.linalg.solve(V, y.astype(float))
