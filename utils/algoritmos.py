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
