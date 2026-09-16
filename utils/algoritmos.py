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
