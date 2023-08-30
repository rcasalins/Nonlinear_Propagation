import numpy as np
import numpy.fft as fft
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns

# ventanas
N = 4096  # Muestreo temporal
T_window = 100 * (10 ** -12)  # Ventana temporal
t_step = T_window / N  # Paso temporal
t_vector = np.arange(-N / 2, N / 2, 1) * t_step  # vector de tiempo
w_window = 2 * np.pi / T_window  # ventana frecuencial
w_step = w_window / N  # paso frecuencial
# w_vector   = np.arange(-N/2,N/2,1) * w_step         # vector de frecuencias
w_vector = (np.pi / T_window) * np.concatenate((np.arange(-(N / 2) + 1, 1, 1), np.arange(0, (N / 2), 1)))

powers = [1, 3, 5, 8, 10]

for ñ in powers:
    # pulso
    P = ñ * (10 ** -3)  # potencia de entrada en W
    c = 3 * (10 ** 8)
    FWHM = 1800 * (10 ** -9)  # Full Width Half Maximum
    width = 1 * (10 ** -9)  # Tamaño del espectro transmitido

    # Pump
    lambda_p = 2100 * (10 ** -9)  # longitud de onda central
    lambda_i_p = lambda_p - (FWHM / 2)  # W inicial
    lambda_f_p = lambda_p + (FWHM / 2)  # W final
    lambda_vector_p = np.linspace(lambda_i_p, lambda_f_p, N)  # vector de lambda

    pulse_aux_pump = np.float128(-((lambda_vector_p - lambda_p) / (width * 2)) ** 2)  # Gaussiana Pump
    pulse_p = np.exp(pulse_aux_pump)  # Pump beam

    # Input field

    pulse = pulse_p
    fourier_pulse = fft.fft(pulse)

    input_field = fourier_pulse / np.max(fourier_pulse)

    # parametros lineales

    # Micro anillo 800 x 400
    # Q_Factor = 5000
    # group_velocity = c/2.696298
    # omega_p = 2*np.pi*c/lambda_p
    # radius = 40 *(10**-6)

    # Micro anillo 600 x 400
    # Q_Factor = 5000
    # group_velocity = c/2.79
    # omega_p = 2*np.pi*c/lambda_p
    # radius = 40 *(10**-6)

    # IE = (Q_Factor*group_velocity)/(omega_p * np.pi * radius)

    device_length = 10 * (10 ** -3)  # largo del dispositivo

    # device_length = 2*np.pi*radius*(IE**2)

    # betas         = np.array([-6.99 * (10 ** -25), -7.27929 * (10 ** -39)])   # parametros de dispersión - beta[0]=s^2/m, beta[1]=s^3/m
    betas = np.array([-6.99 * (10 ** -25), -7.27929 * (10 ** -39)])
    A_eff = 0.96 * (10 ** -12)  # 0.4486 * (10 ** -12)                                         # Area efectiva (m^2)
    Losses = 0  # Perdidas lineales dB/cm
    alpha = 0  # Perdidas lineales en cm^-1

    # parametros no lineales
    n2 = 3 * 4.4 * (
                10 ** -18)  # 10 * (10 ** -18)                                        # Indice de refracción no lineal (m^2)/W
    gamma = (2 * np.pi * n2) / (lambda_p * A_eff)  # Parametro no lineal
    # gamma         = 100

    # integer = 1
    # phi_NL = 2*integer * np.pi
    # P = phi_NL/(device_length*gamma)

    # print(IE)
    # print(device_length)
    # print(gamma)

    # Raman
    # gain_Raman = 8.9 * (10 ** -11)  # Ganancia ramán
    # Gamma_Raman = np.pi * 105  # Ancho de espectral
    # Omega_Raman = 2 * np.pi * 15.6 * (10 ** 12)  # Frecuencia espectral del espectro
    # gamma_Raman = gain_Raman * Gamma_Raman / Omega_Raman
    # tau_1 = 1 / np.sqrt((Omega_Raman ** 2) - (Gamma_Raman ** 2)) * (10 ** 12)
    # tau_2 = (1 / Gamma_Raman) * (10 ** 12)
    # # t = np.linspace(0,.01,10)
    # # yu = np.float128(-t/tau_2)
    # # (np.exp(yu))
    #
    # g = integrate.fixed_quad(
    #     lambda t: (((tau_1 ** 2) + (tau_2 ** 2)) / (tau_1 * tau_2)) * np.sin(t / tau_1) * np.float128(
    #         np.exp(-t / tau_2)), 0, 10000)
    #

    # Especificaciones de la simulación
    step_number = 1000  # pasos de la simulación
    dz = device_length / step_number  # tamaño de paso

    # Operadores
    dispersion_operator = np.exp(1j * (0.5 * betas[0] * (w_vector ** 2) - w_vector) * dz) * np.exp(
        1j * (1 / 6) * betas[1] * (w_vector ** 3) * dz) * np.exp(-alpha / 2 * dz)  # phase factor of the pump wave
    nonlineal_operator = 1j * P * gamma * dz  # Operador no lineal

    # Main Loop
    # Esquema 1/2 nonlinear, dispersion, 1/2 nonlinear
    Initial_HalfStep = input_field * np.exp((abs(input_field) ** 2) * nonlineal_operator / 2)  # 1/2 nonlinear

    for i in range(1, step_number + 1):
        dispersion_HalfStep = fft.ifft(Initial_HalfStep) * dispersion_operator  # dispersion
        field = fft.fft(dispersion_HalfStep)

        Initial_HalfStep = field * np.exp((abs(field) ** 2) * nonlineal_operator / 2)

    field_ssfm = Initial_HalfStep * np.exp((abs(Initial_HalfStep) ** 2) * nonlineal_operator / 2)  #
    final_Field = fft.ifft(field_ssfm)

    # Graficas

    ifft_input_pulse = fft.ifft(input_field) * np.conj(fft.ifft(input_field))
    final_field_dBs = 10 * np.log10((final_Field * np.conj(final_Field)) / np.max(ifft_input_pulse))
    lambdap = 2 * np.pi * c / (w_vector + 2 * np.pi * c / lambda_p) * 10 ** 9
    imput_field_dBs = 10 * np.log10(ifft_input_pulse / np.max(ifft_input_pulse))

    x, y = np.real(lambdap), np.real(final_field_dBs)
    sns.set(style="darkgrid")

    fig, ax = plt.subplots(dpi=100)
    ax.plot(x, y, label="Power =" + str(P), color="#191970")
    # ax.plot(x,imput_field_dBs, color='red')
    ax.fill(x, y, facecolor='#191970', linewidth=1, alpha=0.4)
    plt.ylim([-100, 1])
    plt.xlim([2090, 2110])
    ax.legend()
    plt.show()

    print(P)

    #np.savetxt("powers" + str(ñ) + "SiliconMid-IR", y)
    #np.savetxt("lambdas", x)

print(gamma)
