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



# funcion que crea una matriz A mxn, tiene un seed para poder repetir los resultados
def crearMatriz(m, n, seed=0):
    seed = np.random.seed(seed)
    matriz = np.random.rand(m, n) * 10
    return matriz

# funcion que crea un vector b m con valores aleatorios, toma seed por defecto 0
def crearVector(m, seed=0):
    seed = np.random.seed(seed)
    vector = np.random.rand(m) * 10
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

# funcion F1: costo F1(x) = calcularF(A, b, x) + delta1 * ||x||1
def calcularF1(A, b, x, delta1):
    """
    A: matriz
    b: vector
    x: vector
    delta1: tolerancia
    -o F1: costo
    """
    F1 = calcularF(A, b, x) + delta1 * np.linalg.norm(x, ord=1)
    return F1

# funcion F2: costo F2(x) = calcularF(A, b, x) + delta2 * ||x||2^2
def calcularF2(A, b, x, delta2):
    """
    A: matriz
    b: vector
    x: vector
    delta2: tolerancia
    -o F2: costo
    """
    F2 = calcularF(A, b, x) + delta2 * np.linalg.norm(x, ord=2)**2
    return F2

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

def calcularHessiano(A, funcion, delta=0):
    """
    A: matriz
    -o H: Hessiano de la función
    -o lambdaH: array con autovalores de Hessiano de la función
    nota: solo lo piden con F
    """
    H = 2 * A.T @ A
    lambdaH = np.linalg.eig(H)[0]
    return H, lambdaH

# funcion gradiente: calcula el gradiente de la función a minimizar
def gradiente(A, b, x, funcion, delta=0):
    """
    A: matriz
    b: vector
    x: vector
    funcion: función a minimizar (F, F1 o F2)
    delta: tolerancia (0 por default)
    -o gradiente: gradiente de la función
    nota: no estoy seguro si se pueden hacer las derivadas analíticas o si la idea es usar un método numérico
    """
    if funcion == calcularF:
        return 2 * A.T @ (A @ x - b) # derivada analítica de F
    
    elif funcion == calcularF1:
        return 2 * A.T @ (A @ x - b) + delta * np.sign(x) # derivada analítica de F1
    
    elif funcion == calcularF2:
        return 2 * A.T @ (A @ x - b) + 2 * delta * x # derivada analítica de F2
    

# funcion gradiente descende: algoritmo iterativo para encontrar el mínimo de una función
def gradienteDescendente(A, b, x0, alpha, funcion, delta=0, maxIter=1000):
    """
    A: matriz
    b: vector
    x0: punto inicial
    alpha: tasa de aprendizaje
    funcion: función a minimizar (F, F1 o F2)
    delta: tolerancia (0 por default)
    maxIter: maximo de iteraciones (1000 por default)
    -o x: punto final
    -o F_list: lista con valores de F en cada iteración
    """
    x = x0
    if delta == 0 : F = funcion(A, b, x)
    else : F = funcion(A, b, x, delta)
    F_list = [F]
    
    for i in range(maxIter):
        if delta != 0: 
            x = x - alpha * gradiente(A, b, x, funcion, delta)
            F_new = funcion(A, b, x, delta)
        else:
            x = x - alpha * gradiente(A, b, x, funcion)
            F_new = funcion(A, b, x)

        F_list.append(F_new)
        
        # if abs(F_new - F) < delta: 
        #     break
        # F = F_new
    return x, F_list


# función para resolver el problema de cuadrados mínimos de F utilizando SVD:
def cuadradosMinimos(A, b):
    """
    A: matriz
    b: vector
    -o x: punto final
    fórmula:  x = V.T * Σ^-1 * U.T * b
    nota: esto solo va a funcionar si A es de rango completo (sus valores singulares != 0)
    nota: esto solo funciona para matrices cuadradas (hay que ver que hacemos con esto)
    """
    U, S, Vt = np.linalg.svd(A)
    x = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
    return x

def cuadradosMinimos_MP(A, b):
    """
    A: matriz
    b: vector
    -o x: punto final
    fórmula:  x = V * Σ^+ * U.T * b
    nota: esto va a funcionar incluso si A no es cuadrada
    nota: usa la pseudoinversa de Moore-Penrose para Σ 
    """
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_pinv = np.zeros((U.shape[1], Vt.shape[0]))
    S_pinv[:S.shape[0], :S.shape[0]] = np.diag(1 / S)
    x = Vt.T @ S_pinv @ U.T @ b
    return x

# función para ver si A es de rango completo
def rangoCompleto(A):
    rango = np.linalg.matrix_rank(A)
    if rango == min(A.shape) : return True      # La matriz A es de rango completo
    else : return False         #La matriz A no es de rango completo




m = 5
n = 100
maxIter = 1000

A = crearMatriz(m,n)
b = crearVector(m)
# print("Matriz A: \n", A)
# print("Vector b: \n", b)

sigma, sigmaMax = calcularSigma(A)
print("\nsigmaMax: ", sigmaMax)
H, lambdaH = calcularHessiano(A, calcularF)
lambdaMax = np.max(lambdaH)
print("lambdaMax: ", np.real(lambdaMax))
alfa = 1/lambdaMax
print("alfa: ", alfa)

delta_1 = 1e-3 * lambdaMax
print("delta_1: ", delta_1)
delta_2 = 1e-2 * lambdaMax 
print("delta_2: ", delta_2)

# para F:
x0, F_list_0 = gradienteDescendente(A, b, np.zeros(n), alfa, calcularF, maxIter=maxIter)
# print("\nx0: \n", np.real(x0))
plt.plot(F_list_0, label='F')
plt.xlabel('Iteración')
plt.ylabel('Valor de la función objetivo')
plt.title('Convergencia del gradiente descendente')
plt.show()

# para F1:
x1, F_list_1 = gradienteDescendente(A, b, np.zeros(n), alfa, calcularF1, delta_1, maxIter=maxIter)
# print("\nx1: \n", np.real(x1))
plt.plot(F_list_1, label='F1')
plt.xlabel('Iteración')
plt.ylabel('Valor de la función objetivo')
plt.title('Convergencia del gradiente descendente')
plt.show()

# para F2:
x2, F_list_2 = gradienteDescendente(A, b, np.zeros(n), alfa, calcularF2, delta_2, maxIter=maxIter)
# print("\nx2: \n", np.real(x2))
plt.plot(F_list_2, label='F2')
plt.xlabel('Iteración')
plt.ylabel('Valor de la función objetivo')
plt.title('Convergencia del gradiente descendente')
plt.show()

# para SVD
if rangoCompleto(A):
    # si A es cuadrada uso cuadradosMinimos, sino uso cuadradosMinimos_MP 
    x_svd = cuadradosMinimos(A, b) if A.shape[0] == A.shape[1] else cuadradosMinimos_MP(A, b)
print("\nx_svd: \n", np.real(x_svd))

# comparación de resultados de x0, x1, x2 y x_svd
plt.plot(x0)
plt.title('x0')
plt.show()

plt.plot(x1)
plt.title('x1')
plt.show()

plt.plot(x2)
plt.title('x2')
plt.show()

plt.plot(x_svd)
plt.title('x_svd')
plt.show()
