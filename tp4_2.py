import numpy as np

# Función de costo F(x)
def cost_function(A, b, x):
    return np.linalg.norm(np.dot(A, x) - b)**2

# Algoritmo de descenso por gradiente
def gradient_descent(A, b, s, x_init, max_iterations, target_error):
    x = x_init
    error = cost_function(A, b, x)
    iterations = 0
    
    while error > target_error and iterations < max_iterations:
        gradient = 2 * np.dot(A.T, np.dot(A, x) - b)
        x -= s * gradient
        error = cost_function(A, b, x)
        iterations += 1
    
    return x, iterations

# Número de condición de A
def condition_number(A):
    singular_values = np.linalg.svd(A, compute_uv=False)
    return max(singular_values) / min(singular_values)

# Generar matriz A aleatoria con número de condición dado
def generate_random_matrix(m, n, condition_number):
    U, _, Vt = np.linalg.svd(np.random.randn(m, n), full_matrices=False)
    singular_values = np.linspace(1, condition_number, min(m, n))
    A = np.dot(U * singular_values, Vt)
    return A

# Parámetros
m = 100
n = 100
target_error = 1e-2
max_iterations = 10000
num_condition_values = [1, 10, 100, 1000]  # Valores de número de condición a probar

for condition_value in num_condition_values:
    A = generate_random_matrix(m, n, condition_value)
    b = np.random.randn(m)
    x_init = np.random.randn(n)
    
    Hessian = 2 * np.dot(A.T, A)
    eigenvalues = np.linalg.eigvalsh(Hessian)  # Utilizar eigvalsh para matrices simétricas
    max_eigenvalue = np.max(eigenvalues)
    s = 1 / max_eigenvalue
    
    x_min, iterations = gradient_descent(A, b, s, x_init, max_iterations, target_error)
    predicted_iterations = np.log(target_error) / np.log(1 - s**2)
    
    print("Número de condición de A:", condition_value)
    print("Número de iteraciones (calculado):", iterations)
    print("Número de iteraciones (predicción teórica):", predicted_iterations)
    print("")
