import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter, hilbert
from scipy.interpolate import interp1d
import subprocess
import os
import shutil
from matplotlib.animation import FFMpegWriter

# =============================================================================
# PARÁMETROS GLOBALES
# =============================================================================
BASE_DIR = r"C:\Users\santi\gprmax\gpr_code"
ARCHIVO_BASE = "pec_monostatic.in"
OUT_DIR = "temp_bs_output_mono"
N_TRAZAS = 50
X_START = 0.10
X_END = 0.60
Z_ANT = 0.35                     # altura de la antena (m)
X_IMG_GRID = np.linspace(0.10, 0.60, 200)   # rango en X (m)
Z_GRID = np.linspace(0.05, 0.35, 150)       # rango en Z (altura, m)
V = 299792458                    # velocidad en vacío (m/s)

# Parámetros de filtros - AHORA COINCIDEN CON LA EXCITACIÓN (1-4 GHz)
VENTANA_DEWOW = 11
VENTANA_TIME_ZERO = 30
BANDA_PASA = (1e9, 6e9)          # antes era (1e9, 6e9)

# Parámetros para video y visualización
BLOQUE_TRAZAS = 5
INTERVALO_MS = 200
OUT_VIDEO = "scbp_evolution.mp4"

# =============================================================================
# 1. GENERACIÓN DEL B‑SCAN (sin borrar archivos)
# =============================================================================
def generar_bscan():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Copiar el archivo de excitación al directorio de salida
    #excitation_src = os.path.join(BASE_DIR, "sinc_0_6GHz.txt")
    #excitation_dst = os.path.join(OUT_DIR, "sinc_0_6GHz.txt")
    #if os.path.exists(excitation_src):
    #    shutil.copy2(excitation_src, excitation_dst)
    #    print("Archivo de excitación copiado a", OUT_DIR)
    #else:
    #    print("ERROR: No se encuentra", excitation_src)
    #    exit(1)

    x_positions = np.linspace(X_START, X_END, N_TRAZAS)
    archivos_out = []

    with open(os.path.join(BASE_DIR, ARCHIVO_BASE), 'r') as f:
        plantilla = f.read()

    for i, x_ant in enumerate(x_positions):
        contenido = plantilla.replace('{x_ant}', str(x_ant))
        in_file = os.path.join(OUT_DIR, f"temp_{i:03d}.in")
        with open(in_file, 'w') as f:
            f.write(contenido)

        out_file = in_file.replace('.in', '.out')
        if not os.path.exists(out_file):
            print(f"Ejecutando traza {i+1}/{N_TRAZAS} en x={x_ant:.3f} m...")
            subprocess.run(f"python -m gprMax {in_file}", shell=True, check=True)
        else:
            print(f"Traza {i+1} ya existe, omitiendo simulación.")
        archivos_out.append(out_file)

    return x_positions, archivos_out

# =============================================================================
# 2. CARGA DE DATOS (S11)
# =============================================================================
def cargar_bscan(archivos_out):
    with h5py.File(archivos_out[0], 'r') as f:
        dt = f.attrs['dt']
        nt = f.attrs['Iterations']
        tiempo = np.arange(nt) * dt * 1e9   # nanosegundos

    n_trazas = len(archivos_out)
    data_s11 = np.zeros((nt, n_trazas))

    for i, out_file in enumerate(archivos_out):
        with h5py.File(out_file, 'r') as f:
            Vinc = f['tls']['tl1']['Vinc'][:]
            Vtotal = f['tls']['tl1']['Vtotal'][:]
            Vref = Vtotal - Vinc
            data_s11[:, i] = Vref

    return tiempo, dt, data_s11

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
# 4. SELF-CORRELATION BACK-PROJECTION (SCBP) PROGRESIVO
# =============================================================================
def sc_backprojection_monostatic_progresivo(data, tiempo, x_ant_positions, z_ant,
                                             x_img_grid, z_grid, v, bloque_trazas, callback=None):
    n_ant = len(x_ant_positions)
    n_img_x = len(x_img_grid)
    n_z = len(z_grid)

    # Precalcular interpoladores lineales para cada traza
    interp_funcs = [interp1d(tiempo, data[:, i], kind='linear',
                             bounds_error=False, fill_value=0) for i in range(n_ant)]

    frames = []

    for ia in range(n_ant):
        # Recalcular la imagen completa cada vez que se completa un bloque o es la última traza
        if (ia + 1) % bloque_trazas == 0 or ia == n_ant - 1:
            sub_interp = interp_funcs[:ia+1]
            sub_x_ant = x_ant_positions[:ia+1]
            image_bp_temp = np.zeros((n_img_x, n_z))
            image_sc_temp = np.zeros((n_img_x, n_z))

            # Bucle principal sobre píxeles de la imagen
            for ix, x_img in enumerate(x_img_grid):
                for iz, z in enumerate(z_grid):
                    amplitudes = []
                    for j, x_antj in enumerate(sub_x_ant):
                        dist = np.sqrt((x_antj - x_img)**2 + (z_ant - z)**2)
                        t = 2 * dist / v
                        if 0 < t < tiempo[-1]:
                            amplitudes.append(sub_interp[j](t))
                        else:
                            amplitudes.append(0.0)
                    amps = np.array(amplitudes)
                    suma = np.sum(amps)
                    # Self-correlation term: (sum)^2 - sum of squares
                    sc_term = max(0.0, suma**2 - np.sum(amps**2))
                    image_bp_temp[ix, iz] = suma
                    image_sc_temp[ix, iz] = sc_term

            final_image = image_bp_temp * image_sc_temp
            if np.max(np.abs(final_image)) > 0:
                final_image = final_image / np.max(np.abs(final_image))

            if callback:
                callback(final_image, ia+1)
            frames.append((ia+1, final_image))

    return frames

# =============================================================================
# 5. MIGRACIÓN 1D PARA UNA SOLA TRAZA (A‑scan vs profundidad)
# =============================================================================
def migrar_traza_1d(data_traza, tiempo, x_ant, z_ant, z_grid, v):
    interp = interp1d(tiempo, data_traza, kind='linear', bounds_error=False, fill_value=0)
    profundidades = z_ant - z_grid
    reflectividad = np.zeros_like(profundidades)
    for i, prof in enumerate(profundidades):
        if prof <= 0:
            continue
        t = 2 * prof / v
        if 0 < t < tiempo[-1]:
            reflectividad[i] = interp(t)
    return reflectividad, profundidades

# =============================================================================
# 6. VISUALIZACIÓN EN TIEMPO REAL Y VIDEO
# =============================================================================
def crear_video(frames, x_grid, z_ant, z_grid, output_video, intervalo_ms):
    prof_grid = z_ant - z_grid
    prof_grid_inv = prof_grid[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel('Posición x (m)')
    ax.set_ylabel('Profundidad (m)')
    ax.set_title('Evolución SCBP')
    im = ax.imshow(np.zeros((len(x_grid), len(prof_grid_inv))).T,
                   extent=[x_grid[0], x_grid[-1], prof_grid_inv[-1], prof_grid_inv[0]],
                   origin='upper', cmap='gray', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Amplitud normalizada')
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12,
                   bbox=dict(facecolor='black', alpha=0.6))

    try:
        writer = FFMpegWriter(fps=1000/intervalo_ms, metadata=dict(artist='GPRMax'), bitrate=1800)
        writer.setup(fig, output_video, dpi=100)
        print(f"Generando video {output_video} ...")
    except Exception as e:
        print(f"No se pudo usar FFMpegWriter: {e}. Se guardarán frames como imágenes.")
        writer = None
        frame_dir = "frames_scbp"
        os.makedirs(frame_dir, exist_ok=True)

    for i, (nt, imagen) in enumerate(frames):
        imagen_prof = imagen[:, ::-1]  # invertir Z
        im.set_array(imagen_prof.T)
        text.set_text(f'Trazas usadas: {nt} / {N_TRAZAS}')
        plt.pause(0.01)

        if writer:
            writer.grab_frame()
        else:
            plt.savefig(f"{frame_dir}/frame_{i:04d}.png", dpi=100)

    if writer:
        writer.finish()
        print(f"Video guardado en {output_video}")
    else:
        print(f"Frames guardados en carpeta '{frame_dir}'.")
    plt.close(fig)

# =============================================================================
# 7. SCRIPT PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("=== GENERANDO B‑SCAN MONOESTÁTICO ===")
    x_ant_positions, archivos_out = generar_bscan()

    print("=== CARGANDO DATOS ===")
    tiempo, dt, data_s11 = cargar_bscan(archivos_out)

    print("=== APLICANDO FILTROS ===")
    data_filtrada = aplicar_filtros(data_s11, dt, VENTANA_DEWOW, VENTANA_TIME_ZERO, BANDA_PASA)

    # Envolvente de Hilbert
    data_env = np.abs(hilbert(data_filtrada, axis=0))

    print("=== EJECUTANDO SC-BACKPROJECTION PROGRESIVO ===")
    v_ns = V * 1e-9  # m/ns

    frames = []

    def callback_parcial(imagen, num_trazas):
        frames.append((num_trazas, imagen.copy()))
        # Actualizar la misma figura (sin crear nuevas)
        prof_grid = Z_ANT - Z_GRID
        prof_grid_inv = prof_grid[::-1]
        plt.figure(1)
        plt.clf()
        plt.imshow(imagen[:, ::-1].T,
                   extent=[X_IMG_GRID[0], X_IMG_GRID[-1], prof_grid_inv[-1], prof_grid_inv[0]],
                   origin='upper', cmap='gray')
        plt.colorbar(label='Amplitud')
        plt.xlabel('Posición x (m)')
        plt.ylabel('Profundidad (m)')
        plt.title(f'SCBP - {num_trazas} trazas')
        plt.pause(0.01)

    frames = sc_backprojection_monostatic_progresivo(data_env, tiempo, x_ant_positions, Z_ANT,
                                                      X_IMG_GRID, Z_GRID, v_ns, BLOQUE_TRAZAS, callback_parcial)

    print("=== GENERANDO VIDEO ===")
    crear_video(frames, X_IMG_GRID, Z_ANT, Z_GRID, OUT_VIDEO, INTERVALO_MS)

    print("=== GRAFICANDO RESULTADO FINAL ===")
    if frames:
        num_trazas_final, imagen_final = frames[-1]
        prof_grid = Z_ANT - Z_GRID
        prof_grid_inv = prof_grid[::-1]
        plt.figure(figsize=(8, 6))
        plt.imshow(imagen_final[:, ::-1].T,
                   extent=[X_IMG_GRID[0], X_IMG_GRID[-1], prof_grid_inv[-1], prof_grid_inv[0]],
                   origin='upper', cmap='gray')
        plt.colorbar(label='Amplitud')
        plt.xlabel('Posición x (m)')
        plt.ylabel('Profundidad (m)')
        plt.title(f'SCBP final con {num_trazas_final} trazas')
        plt.show()
    else:
        print("No se generaron frames.")

    # Migración 1D para la traza sobre el cubo (x ≈ 0.15 m)
    idx_cubo = np.argmin(np.abs(x_ant_positions - 0.15))
    traza_cubo = data_env[:, idx_cubo]

    print(f"\n=== MIGRACIÓN 1D PARA TRAZA EN X = {x_ant_positions[idx_cubo]:.3f} m ===")
    reflectividad, profundidades = migrar_traza_1d(traza_cubo, tiempo, x_ant_positions[idx_cubo],
                                                   Z_ANT, Z_GRID, v_ns)

    plt.figure(figsize=(8, 6))
    plt.plot(reflectividad, profundidades, 'b-', linewidth=2)
    plt.xlabel('Amplitud (normalizada)')
    plt.ylabel('Profundidad (m)')
    plt.title(f'A-scan migrado (SCBP) para antena en x={x_ant_positions[idx_cubo]:.3f} m')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.show()

    print("\n¡Proceso completado!")