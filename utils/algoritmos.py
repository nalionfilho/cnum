"""Algoritmos básicos de cálculo numérico."""


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
