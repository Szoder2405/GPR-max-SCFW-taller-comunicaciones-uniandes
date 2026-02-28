import os
import subprocess
import h5py
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_DIR = r"C:\Users\santi\gprmax\gpr_code"
ARCHIVOS = [
    "pec_bistatic.in",
    "agua_arena_bistatic.in",
    "diel_bistatic.in",
    
]

dx_dy_dz = 0.002          # tamaño de celda (m) - debe coincidir con el .in
Z0 = 50
SNR_dB = 20                # relación señal/ruido (dB) para S11 y S21
seed = 42

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def ejecutar_simulacion(in_path):
    out_path = in_path.replace('.in', '.out')
    if not os.path.exists(out_path):
        print(f"Ejecutando simulación: {in_path}")
        cmd = f"python -m gprMax {in_path}"
        subprocess.run(cmd, shell=True, check=True)
    else:
        print(f"Archivo de salida ya existe: {out_path}")
    return out_path

def cargar_datos_tl(out_path):
    with h5py.File(out_path, 'r') as f:
        dt = f.attrs['dt']
        nt = f.attrs['Iterations']
        tiempo = np.arange(nt) * dt * 1e9  # ns

        Vinc_tx = f['tls']['tl1']['Vinc'][:]
        Vtotal_tx = f['tls']['tl1']['Vtotal'][:]
        Vref_tx = Vtotal_tx - Vinc_tx

        Vinc_rx = f['tls']['tl2']['Vinc'][:]
        Vtotal_rx = f['tls']['tl2']['Vtotal'][:]

    return dt, tiempo, Vinc_tx, Vref_tx, Vtotal_rx

def calcular_espectros_complejos(Vinc, Vref, Vrx, dt):
    """Devuelve frecuencia (Hz) y los espectros complejos de S11 y S21."""
    nt = len(Vinc)
    freq = np.fft.fftfreq(nt, d=dt)[:nt//2]

    Vinc_f = np.fft.fft(Vinc)[:nt//2]
    Vref_f = np.fft.fft(Vref)[:nt//2]
    Vrx_f  = np.fft.fft(Vrx)[:nt//2]

    eps = 1e-12
    S11_complex = Vref_f / (Vinc_f + eps)
    S21_complex = Vrx_f  / (Vinc_f + eps)

    return freq, S11_complex, S21_complex

def anadir_ruido_frecuencia(S_complex, SNR_dB, seed=None):
    """
    Añade ruido blanco complejo a un espectro de parámetro S.
    La potencia del ruido se ajusta para que la SNR (en dB) se cumpla respecto a la media de |S|.
    SNR_dB = 10*log10( P_senal_max / P_ruido )
    """
    if seed is not None:
        np.random.seed(seed)

    # Potencia lineal de la señal (módulo al cuadrado)
    potencia_senal = np.abs(S_complex)**2
    pico_potencia = np.mean(potencia_senal)          # potencia media

    if pico_potencia == 0:
        return S_complex

    # Potencia del ruido deseada
    P_ruido = pico_potencia / (10**(SNR_dB/10))

    # Generar ruido complejo (parte real e imaginaria independientes)
    # La varianza de cada parte debe ser P_ruido/2 para que la potencia total sea P_ruido
    sigma = np.sqrt(P_ruido / 2)
    ruido = sigma * (np.random.randn(len(S_complex)) + 1j * np.random.randn(len(S_complex)))

    return S_complex + ruido

def graficar_resultados(tiempo, freq,
                        Vinc, Vref, Vrx,
                        S11_complex, S21_complex,
                        S11_noisy_complex, S21_noisy_complex,
                        titulo_base):
    """Genera una figura con A‑scans (limpios) y parámetros S (con y sin ruido)."""
    # Convertir a dB para graficar
    S11_clean_dB = 20 * np.log10(np.abs(S11_complex) + 1e-12)
    S21_clean_dB = 20 * np.log10(np.abs(S21_complex) + 1e-12)
    S11_noisy_dB = 20 * np.log10(np.abs(S11_noisy_complex) + 1e-12)
    S21_noisy_dB = 20 * np.log10(np.abs(S21_noisy_complex) + 1e-12)

    # Aumentamos ligeramente la altura para dar más espacio al título
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(titulo_base, fontsize=14)

    # A‑scans (señales temporales limpias)
    plt.subplot(2,2,1)
    plt.plot(tiempo, Vref, 'b-', label='Reflejada (limpia)')
    plt.xlabel('Tiempo (ns)')
    plt.ylabel('Voltaje (V)')
    plt.title('Señal reflejada (S11)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(tiempo, Vrx, 'b-', label='Recibida (limpia)')
    plt.xlabel('Tiempo (ns)')
    plt.ylabel('Voltaje (V)')
    plt.title('Señal recibida (S21)')
    plt.grid(True)
    plt.legend()

    # Parámetros S en frecuencia
    plt.subplot(2,2,3)
    plt.plot(freq/1e9, S11_clean_dB, 'b-', label='S11 limpio')
    plt.plot(freq/1e9, S11_noisy_dB, 'r-', alpha=0.7, label='S11 con ruido')
    plt.xlabel('Frecuencia (GHz)')
    plt.ylabel('S11 (dB)')
    plt.title('Coeficiente de reflexión')
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(freq/1e9, S21_clean_dB, 'b-', label='S21 limpio')
    plt.plot(freq/1e9, S21_noisy_dB, 'r-', alpha=0.7, label='S21 con ruido')
    plt.xlabel('Frecuencia (GHz)')
    plt.ylabel('S21 (dB)')
    plt.title('Transmisión')
    plt.grid(True)
    plt.legend()

    # Ajustamos el layout dejando un 5% superior para el título
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# =============================================================================
# PROCESAMIENTO PRINCIPAL
# =============================================================================
np.random.seed(seed)   # para reproducibilidad global
figuras = []

for archivo_in in ARCHIVOS:
    print(f"\n{'='*60}")
    print(f"Procesando: {archivo_in}")
    print('='*60)

    in_path = os.path.join(BASE_DIR, archivo_in)
    if not os.path.exists(in_path):
        print(f"¡Archivo .in no encontrado: {in_path}")
        continue

    out_path = ejecutar_simulacion(in_path)

    # Cargar datos de líneas de transmisión
    dt, tiempo, Vinc_tx, Vref_tx, Vtotal_rx = cargar_datos_tl(out_path)

    # Calcular espectros complejos de S11 y S21 (limpios)
    freq, S11c, S21c = calcular_espectros_complejos(Vinc_tx, Vref_tx, Vtotal_rx, dt)

    # Añadir ruido en frecuencia a cada parámetro S por separado
    # Usamos semillas diferentes para S11 y S21 (aunque no es obligatorio)
    S11_noisy = anadir_ruido_frecuencia(S11c, SNR_dB, seed)
    S21_noisy = anadir_ruido_frecuencia(S21c, SNR_dB, seed+1)

    # Graficar
    titulo = archivo_in.replace('.in', '').replace('_', ' ').title()
    fig = graficar_resultados(tiempo, freq,
                              Vinc_tx, Vref_tx, Vtotal_rx,
                              S11c, S21c,
                              S11_noisy, S21_noisy,
                              titulo)
    figuras.append(fig)

# Mostrar todas las figuras juntas
plt.show()

print("\n¡Procesamiento completado!")