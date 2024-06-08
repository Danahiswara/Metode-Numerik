import time
import numpy as np

def riemann_integral(f, a, b, N):
    width = (b - a) / N
    total = 0.0
    for i in range(N):
        total += f(a + i * width)
    return total * width

def f(x):
    return 4 / (1 + x**2)

# Nilai referensi pi
pi_ref = 3.14159265358979323846

# Variasi nilai N
N_values = [10, 100, 1000, 10000]

# Menyimpan hasil untuk dokumentasi
results = []

for N in N_values:
    start_time = time.time()
    pi_estimate = riemann_integral(f, 0, 1, N)
    end_time = time.time()
    
    error = np.sqrt((pi_estimate - pi_ref) ** 2)
    exec_time = end_time - start_time
    
    results.append((N, pi_estimate, error, exec_time))
    print(f"N={N}, Pi Estimate={pi_estimate}, Error={error}, Time={exec_time} sec")
