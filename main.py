import numpy as np
import math
import numpy.fft as fft
import matplotlib.pyplot as plt
import seaborn as sns

# Temporal Sampling

N = 4096  # Number of points
T_window = 100 * (10 ** -12)  # Temporal window
t_step = T_window / N  # Temporal Step
t_vector = np.arange(-N / 2, N / 2, 1) * t_step  # Time vector

# Frequency Sampling

w_window = 2 * np.pi / T_window  # Frequency window
w_step = w_window / N  # Frequency steps

w_vector = (np.pi / T_window) * np.concatenate(
    (np.arange(-(N / 2) + 1, 1, 1), np.arange(0, (N / 2), 1)))  # Frequency vector

# Wavelength Sampling

lambda_window = 1800 * (10 ** -9)  # Sampling wavelength window
FWHM = 50 * (10 ** -9)  # Full width at half maximum of the pulse in m
lambda_p = 1550 * (10 ** -9)  # Central wavelength in m
lambda_i_p = lambda_p - (lambda_window / 2)  # sampling wavelength initial
lambda_f_p = lambda_p + (lambda_window / 2)  # sampling wavelength final

lambda_vector_p = np.linspace(lambda_i_p, lambda_f_p, N)  # Wavelength vector

# Pulse Parameters

power = 10  # 1000 * (10 ** -3)  # Pump power in W
c = 3 * (10 ** 8)  # Speed of light in vacuum in m/s

pulse_aux_pump = np.float128(-((lambda_vector_p - lambda_p) / (FWHM * 2)) ** 2)  # Gaussian Pump
pulse_p = np.exp(pulse_aux_pump)  # Pump beam

# Input field

fourier_pulse = fft.fft(pulse_p)  # Pulse in time domain
input_field = fourier_pulse / np.max(fourier_pulse)  # Normalized input field

# Device Parameters

device_length = 0.01  # 10 * (10 ** -3)  # Waveguide/Fiber length
betas = np.array([-1.1830 * (10 ** -26), 8.1038 * (10 ** -41)])  # Dispersion parameters. beta2, beta3,... in s^i/m
A_eff = 0.96 * (10 ** -12)  # Effective mode area in m^2
Losses = 0  # Linear losses in dB/cm
alpha = 0  # Linear losses in cm^-1

n2 = 4.4 * (10 ** -18)  # Nonlinear refractive index in m^2/W
gamma = 10  # (2 * np.pi * n2) / (lambda_p * A_eff)

# Spatial sampling

step_number = 1000  # Number of steps for the Split Step Fourier Method (SSFM)
dz = device_length / step_number  # Step size


# Operators

def dispersion_parameters_series(frequency_vector, dispersions):
    taylor_expansion = 0

    for m in range(len(dispersions)):
        aux = dispersions[m] * (1j ** (m + 3) / math.factorial(m + 2)) * ((1j * frequency_vector) ** (m + 2))
        taylor_expansion = taylor_expansion + aux

    return np.array(taylor_expansion)


dispersion_operator = np.exp(dispersion_parameters_series(w_vector, betas) * dz)
nonlinear_operator = 1j * power * gamma * dz

# Main Loop
# Simulation Scheme: 1/2 Nonlinear, dispersion, 1/2 Nonlinear

Initial_HalfStep = input_field * np.exp((abs(input_field) ** 2) * nonlinear_operator / 2)  # 1/2 nonlinear

for i in range(1, step_number + 1):
    dispersion_HalfStep = fft.ifft(Initial_HalfStep) * dispersion_operator  # dispersion
    field = fft.fft(dispersion_HalfStep)

    Initial_HalfStep = field * np.exp((abs(field) ** 2) * nonlinear_operator / 2)

field_ssfm = Initial_HalfStep * np.exp((abs(Initial_HalfStep) ** 2) * nonlinear_operator / 2)  #
final_Field = fft.ifft(field_ssfm)

# Graphics

ifft_input_pulse = fft.ifft(input_field) * np.conj(fft.ifft(input_field))
final_field_dBs = 10 * np.log10((final_Field * np.conj(final_Field)) / np.max(ifft_input_pulse))
lambdap = 2 * np.pi * c / (w_vector + 2 * np.pi * c / lambda_p) * 10 ** 9
imput_field_dBs = 10 * np.log10(ifft_input_pulse / np.max(ifft_input_pulse))

x, y = np.real(lambdap), np.real(final_field_dBs)
sns.set(style="darkgrid")

fig, ax = plt.subplots(dpi=100)
ax.plot(x, y, label="Power =" + str(power), color="#191970")

ax.fill(x, y, facecolor='#191970', linewidth=1, alpha=0.4)
plt.ylim([-100, 1])
plt.xlim([1500, 1600])
ax.legend()
plt.show()
