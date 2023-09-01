import numpy as np
import math

dispersion_parameters = [1, 2, 3]
w_vector = np.array([1, 2, 3])


def dispersion_parameters_series(frequency_vector, dispersions):
    taylor_expansion = 0

    for m in range(len(dispersions)):
        aux = dispersions[m] * (1j ** (m + 3) / math.factorial(m + 2)) * ((1j * frequency_vector) ** (m + 2))
        taylor_expansion = taylor_expansion + aux

    return np.array(taylor_expansion)


dz = 1

dispersion_operator = np.exp(dispersion_parameters_series(w_vector, dispersion_parameters) * dz)

print(dispersion_operator)
