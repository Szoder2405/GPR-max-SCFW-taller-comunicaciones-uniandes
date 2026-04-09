import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter, hilbert
from scipy.interpolate import interp1d
import subprocess
import os

# =============================================================================
# PARÁMETROS GLOBALES
# =============================================================================
BASE_DIR = r"C:\Users\santi\gprmax\gpr_code"
ARCHIVO_BASE = "pec_cscan_single.in"
OUT_DIR = "temp_cscan_output"

# Malla de posiciones de antena
N_X = 30          # número de posiciones en X
N_Y = 30          # número de posiciones en Y
X_START, X_END = 0.15, 0.45   # rango en X (m)
Y_START, Y_END = 0.15, 0.45   # rango en Y (m)
x_positions = np.linspace(X_START, X_END, N_X)
y_positions = np.linspace(Y_START, Y_END, N_Y)

# Altura fija de la antena
Z_ANT = 0.35

# Malla de la imagen 3D (coordenadas absolutas)
X_IMG_GRID = np.linspace(0.10, 0.60, 80)   # rango en X (m)
Y_IMG_GRID = np.linspace(0.10, 0.60, 80)   # rango en Y (m)
Z_GRID = np.linspace(0.05, 0.35, 60)       # rango en Z (altura, m)

V = 299792458                    # velocidad en vacío (m/s)

# Parámetros de filtros
VENTANA_DEWOW = 11
VENTANA_TIME_ZERO = 30
BANDA_PASA = (1e9, 6e9)

# =============================================================================
# 1. GENERACIÓN DEL C‑SCAN (simulaciones)
# =============================================================================
def generar_cscan():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    archivos_out = []
    with open(os.path.join(BASE_DIR, ARCHIVO_BASE), 'r') as f:
        plantilla = f.read()

    total = N_X * N_Y
    idx = 0
    for ix, x_ant in enumerate(x_positions):
        for iy, y_ant in enumerate(y_positions):
            contenido = plantilla.replace('{x_ant}', str(x_ant)).replace('{y_ant}', str(y_ant))
            in_file = os.path.join(OUT_DIR, f"temp_{ix:03d}_{iy:03d}.in")
            with open(in_file, 'w') as f:
                f.write(contenido)

            out_file = in_file.replace('.in', '.out')
            if not os.path.exists(out_file):
                idx += 1
                print(f"Ejecutando simulación {idx}/{total} en (x={x_ant:.3f}, y={y_ant:.3f})...")
                subprocess.run(f"python -m gprMax {in_file}", shell=True, check=True)
            else:
                print(f"Simulación {ix:03d}_{iy:03d} ya existe, omitiendo.")
            archivos_out.append(out_file)
    return archivos_out

# =============================================================================
# 2. CARGA DE DATOS (S11)
# =============================================================================
def cargar_cscan(archivos_out):
    # Obtener dt, nt del primer archivo
    with h5py.File(archivos_out[0], 'r') as f:
        dt = f.attrs['dt']
        nt = f.attrs['Iterations']
        tiempo = np.arange(nt) * dt * 1e9   # ns

    n_trazas = len(archivos_out)
    data = np.zeros((nt, n_trazas))
    for i, out_file in enumerate(archivos_out):
        with h5py.File(out_file, 'r') as f:
            Vinc = f['tls']['tl1']['Vinc'][:]
            Vtotal = f['tls']['tl1']['Vtotal'][:]
            Vref = Vtotal - Vinc
            data[:, i] = Vref
    return tiempo, dt, data

# =============================================================================
# 3. FILTRADO (dewow, time-zero, background removal, band-pass)
# =============================================================================
def aplicar_filtros(data, dt, ventana_dewow, ventana_tz, banda_pasa):
    if ventana_dewow > 0 and ventana_dewow % 2 == 1:
        data = data - savgol_filter(data, ventana_dewow, 1, axis=0)

    n_samples, n_traces = data.shape
    data_tz = np.zeros_like(data)
    for i in range(n_traces):
        ventana = data[:ventana_tz, i]
        idx_peak = np.argmax(np.abs(ventana))
        data_tz[:, i] = np.roll(data[:, i], -idx_peak)

    data_bg = data_tz - np.mean(data_tz, axis=1, keepdims=True)

    freq = fftfreq(data_bg.shape[0], d=dt)
    mask = (np.abs(freq) >= banda_pasa[0]) & (np.abs(freq) <= banda_pasa[1])
    data_f = fft(data_bg, axis=0)
    data_f[~mask] = 0
    data_filtrada = np.real(ifft(data_f, axis=0))

    return data_filtrada

# =============================================================================
# 4. SCBP FINAL (sin progresión)
# =============================================================================
def scbp_3d_final(data, tiempo, x_ant, y_ant, z_ant, x_img, y_img, z_img, v):
    """
    Calcula el volumen final usando self‑correlation back‑projection con todas las antenas.
    Muestra una barra de progreso en la consola.
    """
    n_ant = len(x_ant)
    nx = len(x_img)
    ny = len(y_img)
    nz = len(z_img)

    # Precalcular interpoladores lineales para cada traza
    interp_funcs = [interp1d(tiempo, data[:, i], kind='linear', bounds_error=False, fill_value=0) for i in range(n_ant)]

    image_bp = np.zeros((nx, ny, nz))
    image_sc = np.zeros((nx, ny, nz))

    total_voxels = nx * ny * nz
    processed = 0
    last_percent = -1

    print("Calculando imagen SCBP 3D...")
    for ix, xp in enumerate(x_img):
        for iy, yp in enumerate(y_img):
            for iz, zp in enumerate(z_img):
                amps = []
                for j in range(n_ant):
                    dist = np.sqrt((x_ant[j] - xp)**2 + (y_ant[j] - yp)**2 + (z_ant - zp)**2)
                    t = 2 * dist / v
                    if 0 < t < tiempo[-1]:
                        amps.append(interp_funcs[j](t))
                    else:
                        amps.append(0.0)
                amps = np.array(amps)
                suma = np.sum(amps)
                sc = max(0.0, suma**2 - np.sum(amps**2))
                image_bp[ix, iy, iz] = suma
                image_sc[ix, iy, iz] = sc
                processed += 1
                percent = int(100 * processed / total_voxels)
                if percent > last_percent:
                    last_percent = percent
                    # Barra de progreso de 20 caracteres
                    bar_length = 40
                    filled = int(bar_length * processed // total_voxels)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\rProgreso: |{bar}| {percent}% ({processed}/{total_voxels} vóxeles)", end='', flush=True)
    print()  # nueva línea al terminar

    final_image = image_bp * image_sc
    if np.max(np.abs(final_image)) > 0:
        final_image = final_image / np.max(np.abs(final_image))
    return final_image

# =============================================================================
# 5. SCRIPT PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("=== VERIFICANDO SIMULACIONES ===")
    archivos_out = generar_cscan()

    print("=== CARGANDO DATOS ===")
    tiempo, dt, data = cargar_cscan(archivos_out)

    print("=== APLICANDO FILTROS ===")
    data_filt = aplicar_filtros(data, dt, VENTANA_DEWOW, VENTANA_TIME_ZERO, BANDA_PASA)

    # Envolvente de Hilbert
    data_env = np.abs(hilbert(data_filt, axis=0))

    # Construir listas de coordenadas de antena (en el mismo orden que archivos_out)
    x_ant_list = []
    y_ant_list = []
    for ix, x in enumerate(x_positions):
        for iy, y in enumerate(y_positions):
            x_ant_list.append(x)
            y_ant_list.append(y)
    x_ant = np.array(x_ant_list)
    y_ant = np.array(y_ant_list)

    print("=== EJECUTANDO SCBP 3D FINAL ===")
    v_ns = V * 1e-9          # velocidad en m/ns (el tiempo está en ns)
    volumen_final = scbp_3d_final(data_env, tiempo, x_ant, y_ant, Z_ANT,
                                  X_IMG_GRID, Y_IMG_GRID, Z_GRID, v_ns)

    # Mostrar corte final (profundidad del objeto, por ejemplo 0.20 m desde la antena)
    profundidad_objeto = 0.20   # m (objeto a 0.15 m, antena a 0.35 m)
    idx_z = np.argmin(np.abs((Z_ANT - Z_GRID) - profundidad_objeto))
    z_corte = Z_GRID[idx_z]
    print(f"Mostrando corte en Z = {z_corte:.3f} m (profundidad {Z_ANT - z_corte:.3f} m)")

    plt.figure(figsize=(8,6))
    plt.imshow(volumen_final[:, :, idx_z].T,
               extent=[X_IMG_GRID[0], X_IMG_GRID[-1], Y_IMG_GRID[0], Y_IMG_GRID[-1]],
               origin='lower', cmap='gray')
    plt.colorbar(label='Amplitud normalizada')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'SCBP final - corte a profundidad {profundidad_objeto:.2f} m')
    plt.show()

    # Opcional: guardar el volumen completo en un archivo .npy
    np.save("volumen_scbp_final.npy", volumen_final)
    print("\n¡Proceso completado! Resultado final mostrado.")