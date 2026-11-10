"""
Experimentos: formación de los niveles de la jerarquía en Ruge-Stuben (PyAMG)
=============================================================================

Instalación: python -m pip install pyamg       

Experimentos:
    1. Tamaño de cada nivel para n = 8, 12, 16, 20
    2. Separacion C/F y matriz de fuerza de conexion
    3. Operadores P y R, y verificacion de Galerkin (A_{k+1} = R A_k P)
    4. Proyección del vector b de un nivel al siguiente
    5. Variaciones: umbral theta, interpolacion y segunda pasada de RS
    6. Mismos tamaños, pero con Poisson 2D

Nota sobre MAX_COARSE: con el valor por defecto de PyAMG (10), una matriz de
tamaño 8 no genera ningun nivel adicional. Por eso aqui se usa un valor bajo.
"""
import sys

import numpy as np
import pyamg
from pyamg.gallery import poisson

np.set_printoptions(precision=2, suppress=True, linewidth=170)

TAMANOS = (8, 12, 16, 20)   # tamañs de matriz a estudiar
MAX_COARSE = 3              # tamaño maximo permitido en el nivel mas grueso


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def titulo(texto, nivel=1):
    """Imprime un encabezado para separar la salida."""
    barra = ("=" if nivel == 1 else "-") * 78
    print(f"\n{barra}\n{texto}\n{barra} - experimentos_ruge_stuben.py:36")


def construir(A, **opciones):
    """
    Construye la jerarquia Ruge-Stuben.
    keep=True conserva la matriz de fuerza de conexion C en cada nivel.
    Cualquier opcion extra (strength, CF, interpolation...) se pasa a PyAMG.
    """
    return pyamg.ruge_stuben_solver(A, max_coarse=MAX_COARSE, keep=True, **opciones)


def tamanos(ml):
    """Tamaño (numero de incognitas) de la matriz de cada nivel."""
    return [lvl.A.shape[0] for lvl in ml.levels]


def pares_de_niveles(ml):
    """Recorre (k, nivel k, nivel k+1). El ultimo nivel no tiene P, R ni splitting."""
    for k, lvl in enumerate(ml.levels[:-1]):
        yield k, lvl, ml.levels[k + 1]


def indices_C_F(splitting, n_grueso):
    """
    Devuelve los indices de los puntos C y de los puntos F.
    Se asume True = punto C. Si el conteo no cuadra con el tamaño del nivel
    siguiente, se prueba la convencion contraria. (El Experimento 3 verifica
    la convencion: las filas de P en los puntos C deben ser identidad.)
    """
    s = np.asarray(splitting).astype(bool)
    if s.sum() == n_grueso:
        marca_C = s
    elif (~s).sum() == n_grueso:
        marca_C = ~s
    else:
        raise ValueError("El numero de puntos C no coincide con el tamaño del nivel siguiente")
    return np.where(marca_C)[0], np.where(~marca_C)[0]


# ----------------------------------------------------------------------------
# Experimento 1: tamaño de cada nivel
# ----------------------------------------------------------------------------
def experimento_1():
    titulo("EXPERIMENTO 1: tamaño de cada nivel (Poisson 1D)")
    for n in TAMANOS:
        ml = construir(poisson((n,), format="csr"))
        print(f"\nn = {n} - experimentos_ruge_stuben.py:83")
        print(ml)  # resumen de PyAMG: tamaño y no-ceros por nivel
        print("Tamaños por nivel: - experimentos_ruge_stuben.py:85", tamanos(ml))


# ----------------------------------------------------------------------------
# Experimento 2: separacion C/F y fuerza de conexion
# ----------------------------------------------------------------------------
def experimento_2():
    titulo("EXPERIMENTO 2: separacion C/F y matriz de fuerza de conexion")
    for n in TAMANOS:
        ml = construir(poisson((n,), format="csr"))
        titulo(f"n = {n}", 2)
        for k, lvl, sig in pares_de_niveles(ml):
            n_k, n_sig = lvl.A.shape[0], sig.A.shape[0]
            C_idx, F_idx = indices_C_F(lvl.splitting, n_sig)
            print(f"\nNivel {k} (tamano {n_k})  >  Nivel {k + 1} (tamano {n_sig}) - experimentos_ruge_stuben.py:99")
            print("Matriz de fuerza de conexion C (entradas distintas de cero = conexiones fuertes): - experimentos_ruge_stuben.py:100")
            print(lvl.C.toarray())
            print("Puntos C: - experimentos_ruge_stuben.py:102", C_idx)
            print("Puntos F: - experimentos_ruge_stuben.py:103", F_idx)
            print(f"Cantidad de puntos C = {len(C_idx)}  |  tamano del nivel siguiente = {n_sig} - experimentos_ruge_stuben.py:104")


# ----------------------------------------------------------------------------
# Experimento 3: operadores P y R, Galerkin
# ----------------------------------------------------------------------------
def experimento_3():
    titulo("EXPERIMENTO 3: operadores P y R, y producto de Galerkin")
    for n in TAMANOS:
        ml = construir(poisson((n,), format="csr"))
        titulo(f"n = {n}", 2)
        for k, lvl, sig in pares_de_niveles(ml):
            C_idx, _ = indices_C_F(lvl.splitting, sig.A.shape[0])
            P = lvl.P.toarray()
            R = lvl.R.toarray()
            A_sig = sig.A.toarray()

            print(f"\nNivel {k} > {k + 1} - experimentos_ruge_stuben.py:121")
            print(f"P_{k} tiene forma {P.shape} (filas = nivel fino, columnas = nivel grueso) - experimentos_ruge_stuben.py:122")
            print(f"P_{k} =\n{P} - experimentos_ruge_stuben.py:123")
            print(f"R_{k} =\n{R} - experimentos_ruge_stuben.py:124")
            print(f"A_{k + 1} =\n{A_sig} - experimentos_ruge_stuben.py:125")

            # Verificaciones (se imprimen True/False; no detienen el programa)
            print("Verificaciones: - experimentos_ruge_stuben.py:128")
            print("R = P^T: - experimentos_ruge_stuben.py:129", np.allclose(R, P.T))
            print("R A P == A del siguiente nivel: - experimentos_ruge_stuben.py:130",
                  np.allclose((lvl.R @ lvl.A @ lvl.P).toarray(), A_sig))
            print("Filas de P en puntos C = identidad: - experimentos_ruge_stuben.py:132", np.allclose(P[C_idx, :], np.eye(len(C_idx))))
            print("A del siguiente nivel es simetrica: - experimentos_ruge_stuben.py:133", np.allclose(A_sig, A_sig.T))
            print("Suma de cada fila de P: - experimentos_ruge_stuben.py:134", P.sum(axis=1))


# ----------------------------------------------------------------------------
# Experimento 4: proyeccion del vector b
# ----------------------------------------------------------------------------
def experimento_4():
    titulo("EXPERIMENTO 4: proyeccion del vector b entre niveles")
    for n in TAMANOS:
        ml = construir(poisson((n,), format="csr"))
        titulo(f"n = {n}", 2)
        b_k = np.arange(1, n + 1, dtype=float)   # b = 1, 2, ..., n (facil de rastrear)
        print("b_0 = - experimentos_ruge_stuben.py:146", b_k)
        for k, lvl, sig in pares_de_niveles(ml):
            b_sig = lvl.R @ b_k                  # b_{k+1} = R_k b_k = P_k^T b_k
            P = lvl.P.toarray()
            print(f"\nNivel {k} > {k + 1}:  b_{k + 1} = R_{k} b_{k}   (tamano {len(b_k)} > {len(b_sig)}) - experimentos_ruge_stuben.py:150")
            for j in range(P.shape[1]):
                filas = np.nonzero(P[:, j])[0]
                terminos = " + ".join(f"{P[i, j]:.2f}*{b_k[i]:.0f}" for i in filas)
                print(f"b_{k + 1}[{j}] = {terminos} = {b_sig[j]:.2f} - experimentos_ruge_stuben.py:154")
            print(f"b_{k + 1} = - experimentos_ruge_stuben.py:155", b_sig)
            print("Coincide con P^T b: - experimentos_ruge_stuben.py:156", np.allclose(b_sig, P.T @ b_k))
            b_k = b_sig


# ----------------------------------------------------------------------------
# Experimento 5: variaciones de los parametros
# ----------------------------------------------------------------------------
def experimento_5(n=16):
    titulo(f"EXPERIMENTO 5: variaciones de parametros (Poisson 1D, n = {n})")
    configuraciones = {
        "base (theta=0.25, clasica, RS)": {},
        "theta = 0.5": dict(strength=("classical", {"theta": 0.5})),
        "theta = 0.9": dict(strength=("classical", {"theta": 0.9})),
        "interpolacion directa": dict(interpolation="direct"),
        "RS con segunda pasada": dict(CF=("RS", {"second_pass": True})),
    }
    A = poisson((n,), format="csr")
    for nombre, opciones in configuraciones.items():
        ml = construir(A, **opciones)
        nnz = [lvl.A.nnz for lvl in ml.levels]
        complejidad = sum(nnz) / nnz[0]
        print(f"\n{nombre} - experimentos_ruge_stuben.py:177")
        print("tamanos por nivel : - experimentos_ruge_stuben.py:178", tamanos(ml))
        print("noceros por nivel: - experimentos_ruge_stuben.py:179", nnz)
        print(f"complejidad del operador: {complejidad:.2f} - experimentos_ruge_stuben.py:180")
        # Puntos C elegidos en el primer nivel
        if len(ml.levels) > 1:
            C_idx, _ = indices_C_F(ml.levels[0].splitting, ml.levels[1].A.shape[0])
            print("puntos C del nivel 0: - experimentos_ruge_stuben.py:184", C_idx)


# ----------------------------------------------------------------------------
# Experimento 6: Poisson 2D con los mismos tamaños
# ----------------------------------------------------------------------------
def experimento_6():
    titulo("EXPERIMENTO 6: Poisson 2D (8 = 4x2, 12 = 4x3, 16 = 4x4, 20 = 5x4)")
    mallas = ((4, 2), (4, 3), (4, 4), (5, 4))
    for malla in mallas:
        A = poisson(malla, format="csr")
        ml = construir(A)
        print(f"\nMalla {malla[0]}x{malla[1]}  (n = {A.shape[0]}) - experimentos_ruge_stuben.py:196")
        print("tamaños por nivel: - experimentos_ruge_stuben.py:197", tamanos(ml))
        for k, lvl, sig in pares_de_niveles(ml):
            C_idx, F_idx = indices_C_F(lvl.splitting, sig.A.shape[0])
            print(f"nivel {k}: C = {C_idx}  |  F = {F_idx} - experimentos_ruge_stuben.py:200")


# ----------------------------------------------------------------------------
EXPERIMENTOS = {
    1: experimento_1,
    2: experimento_2,
    3: experimento_3,
    4: experimento_4,
    5: experimento_5,
    6: experimento_6,
}

if __name__ == "__main__":
    elegidos = [int(a) for a in sys.argv[1:]] or list(EXPERIMENTOS)
    for numero in elegidos:
        EXPERIMENTOS[numero]()
