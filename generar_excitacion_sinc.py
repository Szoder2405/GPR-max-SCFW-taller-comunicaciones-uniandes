import numpy as np

dx = 0.008
c = 299792458
dt = dx / (c * np.sqrt(3))          # 1.54067e-11 s
time_window = 10e-9
nt = 651
t = np.linspace(0, time_window, nt)

# Parámetros del pulso (1-4 GHz)
f_low, f_high = 0.0, 6e9
B = f_high - f_low
f0 = (f_low + f_high) / 2
t0 = time_window / 2

s = np.sinc(B * (t - t0)) * np.cos(2 * np.pi * f0 * (t - t0))
s_windowed = s * np.hanning(nt)
s_norm = s_windowed / np.max(np.abs(s_windowed))

# --- Extender el vector de tiempo y amplitud para cubrir hasta time_window + dt ---
# Añadimos un punto más en t = time_window + dt con amplitud 0
t_ext = np.append(t, time_window + dt)
amp_ext = np.append(s_norm, 0.0)

# Guardar en formato excitation_file (con cabecera)
with open('sinc_0_6GHz.txt', 'w') as f:
    f.write('time mi_pulso\n')
    for ti, amp in zip(t_ext, amp_ext):
        f.write(f'{ti:.12e} {amp:.12e}\n')

print(f"Archivo generado: sinc_0_6GHz.txt con {len(t_ext)} líneas")
print(f"Último tiempo: {t_ext[-1]:.3e} s (mayor que time_window)")