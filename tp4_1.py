import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

"""
1. Generar matrices A y vectores b aleatorios con las dimensiones indicadas en cada consigna

2. Resolver el problema de minimización de F, F2 y F1 utilizando el algoritmo de gradiente descendente con los parámetros indicados en cada consigna

3. Utilizar SVD para obtener la solución y compararla con la obtenida en el paso anterior

4. Analizar los resultados obtenidos y responder las preguntas planteadas en cada consigna

5. Para la segunda consigna, generar matrices aleatorias con un número de condición dado y resolver el problema de minimización de F para distintos números de condición

6. Calcular el número de iteraciones en función del número de condición de A y comparar con la predicción teórica

Algunas definiciones antes de pasar a las consignas: sigma son los valores principales de A, λ son los
autovalores de H, donde H es el Hessiano de F (no de F1 ni F2).
"""

m = 5
n = 100

# funcion que crea una matriz A mxn con valores aleatorios, tiene un seed para poder repetir los resultados
def crearMatriz(m,n, seed=0):
    np.random.seed(seed)
    matriz = np.random.rand(m,n)
    return matriz

# funcion que crea un vector b m con valores aleatorios, toma seed por defecto 0
def crearVector(m, seed=0):
    seed = np.random.seed(seed)
    vector = np.random.rand(m)
    return vector


# funcion F: costo F(x) = (Ax − b).T (Ax − b) 
def calcularF(A, b, x):
    """
    A: matriz
    b: vector
    x: vector
    -o F: costo
    """
    F = (A @ x - b).T @ (A @ x - b)
    return F

# función para calcular valores principales de A
def calcularSigma(A):
    """
    A: matriz
    -o sigma: array con valores principales de A
    -o sigmaMax: valor principal máximo
    """
    sigma = np.linalg.svd(A, compute_uv=False) # compute_uv=False para que no calcule U y V
    sigmaMax = np.max(sigma)
    return sigma, sigmaMax

# función para calcular el Hessiano de F y sus autovalores
def Hessiano(A):
    """
    A: matriz
    -o H: Hessiano de F
    -o H: array con autovalores de Hessiano de F
    """
    H = 2 * A.T @ A
    lambdaH = np.linalg.eig(H)[0]
    return H, lambdaH
    # REVISAR!! HESSIANO (LO HIZO EL COPILOT ;) )


# funcion gradiente descende: algoritmo iterativo para encontrar el mínimo de una función
def gradienteDescendente(A, b, x0, alpha, delta=0, maxIter=1000):
    """
    A: matriz
    b: vector
    x0: punto inicial
    alpha: tasa de aprendizaje
    delta: tolerancia (0 por default)
    maxIter: maximo de iteraciones (1000 por default)
    -o x: punto final

    si delta=0 => se itera hasta alcanzar maxIter
    """
    x = x0
    F = calcularF(A, b, x)
    F_list = [F]
    for i in range(maxIter):
        x = x - alpha * 2 * A.T @ (A @ x - b) # gradiente de F: REVISAR (fórmula 3 del tp)!!
        F = calcularF(A, b, x)
        F_list.append(F)
        if F < delta:
            break
    return x, F_list





A = crearMatriz(m,n)
b = crearVector(m)
# print("Matriz A: \n", A)
# print("Vector b: \n", b)

sigma, sigmaMax = calcularSigma(A)

H, lambdaH = Hessiano(A)
lambdaMax = np.max(lambdaH)

alfa = 1/lambdaMax

# delta_1 = 1e-3 * sigmaMax
# delta_2 = 1e-2 * sigmaMax (estos van recién para el ej 2)

maxIter = 1000

x = gradienteDescendente(A, b, np.zeros(n), alfa) 
