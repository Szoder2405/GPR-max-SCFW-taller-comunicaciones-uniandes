#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCFBP 3D grueso: utiliza todas las trazas (X e Y) para enfocar cada plano X‑Z.
Aceleración SCFBP en la dirección X. Ideal para C‑scan de alta calidad.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter, hilbert
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import subprocess
import os

# =============================================================================
# PARÁMETROS GLOBALES (ajustables)
# =============================================================================
BASE_DIR = r"C:\Users\santi\gprmax\gpr_code"
PLANTILLA = "pec_cscan_single.in"          # plantilla con {x_ant} y {y_ant}
OUT_DIR = "temp_cscan_output"              # carpeta para archivos temporales

# Barrido de antena
X_START, X_END = 0.15, 0.45
Y_START, Y_END = 0.15, 0.45
N_TRAZAS_X = 30
N_TRAZAS_Y = 30
x_positions = np.linspace(X_START, X_END, N_TRAZAS_X)
y_positions = np.linspace(Y_START, Y_END, N_TRAZAS_Y)

# Parámetros físicos
c = 299792458               # velocidad de la luz (m/s)
Z_ANT = 0.35                # altura de la antena (m)
EPS_R = 1.0                 # permitividad relativa (1.0 = vacío)
Z_SURFACE = 0.0             # sin interfaz (vacío)

# Malla de la imagen 3D
X_IMG = np.linspace(0.15, 0.45, 60)   # rango X (m)
Z_IMG = np.linspace(0.00, 0.35, 60)   # rango Z (altura, m)
Y_IMG = np.linspace(0.15, 0.45, 60)   # rango Y (m)
dz = Z_IMG[1] - Z_IMG[0]

# Filtros de pre‑procesamiento
VENTANA_DEWOW = 11
VENTANA_TIME_ZERO = 30
BANDA_PASA = (0.5e9, 4e9)     # rango de frecuencias útil (Hz)

# Parámetros SCFBP
SUBA_PER_SIZE = 8            # número de trazas por sub‑apertura inicial
UPSAMPLE_FACTOR = 2          # factor de upsampling en cada fusión
Y_APERTURE_HALF = 0.03          # <<< NUEVO: semi‑ancho en metros para filtro en Y

# =============================================================================
# 1. FUNCIONES GEOMÉTRICAS (refracción / vacío)
# =============================================================================
def punto_refraccion(x_ant, x_img, z_img, z_ant, z_surf, eps_r):
    def f(xr):
        R1 = np.hypot(xr - x_ant, z_ant - z_surf)
        R2 = np.hypot(x_img - xr, z_surf - z_img)
        if R1 == 0 or R2 == 0:
            return 0.0
        return (xr - x_ant) / R1 - np.sqrt(eps_r) * (x_img - xr) / R2
    x_min = min(x_ant, x_img) - 0.05
    x_max = max(x_ant, x_img) + 0.05
    try:
        xr = brentq(f, x_min, x_max, xtol=1e-6)
    except ValueError:
        xr = (x_ant + x_img) / 2
    return xr

def tiempo_viaje_3d(x_ant, y_ant, x_img, y_img, z_img, z_ant, z_surf, eps_r):
    """Tiempo de ida y vuelta (s) en 3D, con refracción si eps_r != 1."""
    if np.isclose(eps_r, 1.0):
        # Vacío: línea recta 3D
        R = np.sqrt((x_ant - x_img)**2 + (y_ant - y_img)**2 + (z_ant - z_img)**2)
        return 2.0 * R / c
    else:
        # Refracción en el plano vertical que contiene (x_ant, y_ant) y (x_img, y_img)
        # Se reduce a un problema 2D en ese plano
        dx = x_img - x_ant
        dy = y_img - y_ant
        d_horiz = np.hypot(dx, dy)
        if d_horiz == 0:
            # Debajo de la antena
            R1 = z_ant - z_surf
            R2 = z_surf - z_img
            return 2.0 * (R1 / c + R2 / (c / np.sqrt(eps_r)))
        # Coordenadas en el plano de incidencia
        x_local = d_horiz
        # Refracción en 2D
        def f_local(xr):
            R1 = np.hypot(xr, z_ant - z_surf)
            R2 = np.hypot(x_local - xr, z_surf - z_img)
            if R1 == 0 or R2 == 0:
                return 0.0
            return xr / R1 - np.sqrt(eps_r) * (x_local - xr) / R2
        try:
            xr = brentq(f_local, 0, x_local, xtol=1e-6)
        except ValueError:
            xr = x_local / 2
        R1 = np.hypot(xr, z_ant - z_surf)
        R2 = np.hypot(x_local - xr, z_surf - z_img)
        return 2.0 * (R1 / c + R2 / (c / np.sqrt(eps_r)))

# =============================================================================
# 2. GENERACIÓN DE DATOS (C‑scan)
# =============================================================================
def generar_cscan():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    with open(os.path.join(BASE_DIR, PLANTILLA), 'r') as f:
        plantilla = f.read()

    archivos_out = [[None for _ in range(N_TRAZAS_X)] for _ in range(N_TRAZAS_Y)]
    total = N_TRAZAS_X * N_TRAZAS_Y
    contador = 0
    for iy, y_ant in enumerate(y_positions):
        for ix, x_ant in enumerate(x_positions):
            contenido = plantilla.replace('{x_ant}', str(x_ant)).replace('{y_ant}', str(y_ant))
            in_file = os.path.join(OUT_DIR, f"temp_{ix:03d}_{iy:03d}.in")
            out_file = in_file.replace('.in', '.out')
            if os.path.exists(out_file):
                print(f"Traza ({ix+1}/{N_TRAZAS_X}, {iy+1}/{N_TRAZAS_Y}) ya existe, omitiendo.")
            else:
                with open(in_file, 'w') as f:
                    f.write(contenido)
                contador += 1
                print(f"Ejecutando ({contador}/{total}) en x={x_ant:.3f}, y={y_ant:.3f}...")
                subprocess.run(f"python -m gprMax {in_file}", shell=True, check=True)
            archivos_out[iy][ix] = out_file
    return x_positions, y_positions, archivos_out

# =============================================================================
# 3. CARGA Y FILTRADO 3D
# =============================================================================
def cargar_datos_3d(archivos_out):
    with h5py.File(archivos_out[0][0], 'r') as f:
        dt = f.attrs['dt']
        nt = f.attrs['Iterations']
        tiempo = np.arange(nt) * dt * 1e9   # ns

    ny = len(archivos_out)
    nx = len(archivos_out[0])
    data_3d = np.zeros((nt, nx, ny), dtype=np.complex128)

    for iy in range(ny):
        for ix in range(nx):
            out_file = archivos_out[iy][ix]
            with h5py.File(out_file, 'r') as f:
                Vinc = f['tls']['tl1']['Vinc'][:]
                Vtotal = f['tls']['tl1']['Vtotal'][:]
                Vref = Vtotal - Vinc
                data_3d[:, ix, iy] = Vref
    return tiempo, dt, data_3d

def aplicar_filtros_3d(data_3d, dt):
    nt, nx, ny = data_3d.shape
    data_filtrada = np.zeros_like(data_3d, dtype=np.float64)

    for iy in range(ny):
        for ix in range(nx):
            traza = data_3d[:, ix, iy].real
            if VENTANA_DEWOW > 0 and VENTANA_DEWOW % 2 == 1:
                traza = traza - savgol_filter(traza, VENTANA_DEWOW, 1)
            ventana = traza[:VENTANA_TIME_ZERO]
            idx_peak = np.argmax(np.abs(ventana))
            traza = np.roll(traza, -idx_peak)
            freq = fftfreq(nt, d=dt)
            mask = (np.abs(freq) >= BANDA_PASA[0]) & (np.abs(freq) <= BANDA_PASA[1])
            traza_f = fft(traza)
            traza_f[~mask] = 0
            traza = np.real(ifft(traza_f))
            data_filtrada[:, ix, iy] = traza

    # Background removal por cada plano X‑Z
    for iy in range(ny):
        data_filtrada[:, :, iy] -= np.mean(data_filtrada[:, :, iy], axis=1, keepdims=True)

    data_env = np.abs(hilbert(data_filtrada, axis=0))
    return data_env

# =============================================================================
# 4. BACK-PROJECTION 3D PARA UNA SUBAPERTURA (todas las trazas disponibles)
# =============================================================================
def sub_image_bp_3d(traces, x_ant, y_ant, tiempo, x_grid, z_grid, y_plane,
                    z_ant, z_surf, eps_r):
    """
    traces: matriz (nt, n_traces) – todas las trazas de la subapertura.
    x_ant, y_ant: coordenadas de cada traza.
    y_plane: coordenada Y del plano imagen (fija para este llamado).
    Retorna imagen X‑Z (len(x_grid), len(z_grid)).
    """
    nx, nz = len(x_grid), len(z_grid)
    img = np.zeros((nx, nz))
    n_traces = traces.shape[1]
    interp = [interp1d(tiempo, traces[:, i], kind='linear', bounds_error=False, fill_value=0)
              for i in range(n_traces)]

    for ix, xp in enumerate(x_grid):
        for iz, zp in enumerate(z_grid):
            suma = 0.0
            for i in range(n_traces):
                t = tiempo_viaje_3d(x_ant[i], y_ant[i], xp, y_plane, zp,
                                    z_ant, z_surf, eps_r) * 1e9   # a ns
                if 0 < t < tiempo[-1]:
                    suma += interp[i](t)
            img[ix, iz] = suma
    return img

# =============================================================================
# 5. COMPRESIÓN ESPECTRAL (FA1 y FA2)
# =============================================================================
def aplicar_FA1(img, x_grid, z_grid, z_ant, z_surf, eps_r, Kc):
    nx, nz = len(x_grid), len(z_grid)
    eps_prime = (1/np.sqrt(eps_r) - 1)**2
    y_k = z_ant - z_surf
    FA1 = np.zeros((nx, nz), dtype=np.complex128)
    for ix, x in enumerate(x_grid):
        for iz, z in enumerate(z_grid):
            term1 = -Kc * np.sqrt(eps_prime * x**2 + y_k**2)
            term2 = -Kc * np.sqrt(eps_r) * np.sqrt(x**2 / eps_r + z**2)
            term3 = 2 * Kc * z
            phase = term1 + term2 + term3
            FA1[ix, iz] = np.exp(1j * phase)
    return img * FA1

def aplicar_FA2(img_freq_range, x_grid, z_grid, Kc, dK_vec, z_ant, z_surf, eps_r):
    nx, nz = img_freq_range.shape
    eps_prime = (1/np.sqrt(eps_r) - 1)**2
    y_k = z_ant - z_surf
    FA2 = np.zeros((nx, nz), dtype=np.complex128)
    for ix, x in enumerate(x_grid):
        for iz in range(nz):
            z = z_grid[iz]
            dK = dK_vec[iz]
            term1 = -dK * np.sqrt(eps_prime * x**2 + y_k**2)
            term2 = -dK * np.sqrt(eps_r) * np.sqrt(x**2 / eps_r + z**2)
            phase = term1 + term2
            FA2[ix, iz] = np.exp(1j * phase)
    return img_freq_range * FA2

# =============================================================================
# 6. SCFBP 2D COMPLETO (con apertura 3D en el back‑projection)
# =============================================================================
def scfbp_2d_3daperture(traces, x_ant, y_ant, tiempo, x_grid_base, z_grid, y_plane,
                        z_ant, z_surf, eps_r, suba_size, up_factor,
                        return_complex=False, target_x_grid=None):
    """
    SCFBP para un plano Y fijo (y_plane). Utiliza todas las trazas (X,Y) para el BP.
    """
    n_traces = traces.shape[1]
    suba_indices = [list(range(i, min(i+suba_size, n_traces))) for i in range(0, n_traces, suba_size)]
    sub_images = []
    sub_grids_x = []

    print("  Generando sub‑imágenes iniciales...")
    for idxs in suba_indices:
        x_sub = x_ant[idxs]
        y_sub = y_ant[idxs]
        traces_sub = traces[:, idxs]
        img = sub_image_bp_3d(traces_sub, x_sub, y_sub, tiempo, x_grid_base, z_grid, y_plane,
                              z_ant, z_surf, eps_r)
        sub_images.append(img)
        sub_grids_x.append(x_grid_base.copy())

    fc = (BANDA_PASA[0] + BANDA_PASA[1]) / 2
    Kc = 4 * np.pi * fc / c
    ky = fftfreq(len(z_grid), d=dz)
    K = ky / np.sqrt(eps_r)
    dK_vec = K - Kc

    nivel = 0
    while len(sub_images) > 1:
        print(f"  Fusión nivel {nivel+1}, {len(sub_images)} sub‑imágenes")
        new_images = []
        new_grids_x = []
        for k in range(0, len(sub_images), 2):
            if k+1 < len(sub_images):
                imgA, gridA = sub_images[k], sub_grids_x[k]
                imgB, gridB = sub_images[k+1], sub_grids_x[k+1]

                imgA = aplicar_FA1(imgA, gridA, z_grid, z_ant, z_surf, eps_r, Kc)
                imgB = aplicar_FA1(imgB, gridB, z_grid, z_ant, z_surf, eps_r, Kc)

                fftA = fft(imgA, axis=1)
                fftB = fft(imgB, axis=1)

                fftA = aplicar_FA2(fftA, gridA, z_grid, Kc, dK_vec, z_ant, z_surf, eps_r)
                fftB = aplicar_FA2(fftB, gridB, z_grid, Kc, dK_vec, z_ant, z_surf, eps_r)

                fftA_az = fft(fftA, axis=0)
                fftB_az = fft(fftB, axis=0)

                nxA, nz = fftA_az.shape
                nxB = fftB_az.shape[0]
                new_nx = int(max(nxA, nxB) * up_factor)
                fftA_up = np.zeros((new_nx, nz), dtype=np.complex128)
                fftB_up = np.zeros((new_nx, nz), dtype=np.complex128)
                startA = (new_nx - nxA) // 2
                fftA_up[startA:startA+nxA, :] = fftA_az
                startB = (new_nx - nxB) // 2
                fftB_up[startB:startB+nxB, :] = fftB_az

                fft_fused = fftA_up + fftB_up
                img_fused_az = ifft(fft_fused, axis=0)
                img_fused = ifft(img_fused_az, axis=1)

                x_min = min(gridA[0], gridB[0])
                x_max = max(gridA[-1], gridB[-1])
                new_grid = np.linspace(x_min, x_max, new_nx)

                new_images.append(img_fused)
                new_grids_x.append(new_grid)
            else:
                new_images.append(sub_images[k])
                new_grids_x.append(sub_grids_x[k])
        sub_images = new_images
        sub_grids_x = new_grids_x
        nivel += 1

    final_img_complex = sub_images[0]
    final_x_grid = sub_grids_x[0]

    if target_x_grid is not None:
        interp_real = interp1d(final_x_grid, np.real(final_img_complex), axis=0,
                               kind='linear', bounds_error=False, fill_value=0)
        interp_imag = interp1d(final_x_grid, np.imag(final_img_complex), axis=0,
                               kind='linear', bounds_error=False, fill_value=0)
        final_img_complex = interp_real(target_x_grid) + 1j * interp_imag(target_x_grid)

    if return_complex:
        return final_img_complex, None, final_x_grid
    else:
        final_img_mag = np.abs(final_img_complex)
        if np.max(final_img_mag) > 0:
            final_img_mag /= np.max(final_img_mag)
        return final_img_mag, None, final_x_grid

# =============================================================================
# 7. PROCESAMIENTO DEL VOLUMEN COMPLETO (plano por plano Y)
# =============================================================================
def procesar_volumen_3d(data_3d_filt, x_ant_pos, y_ant_pos, tiempo,
                        X_IMG, Y_IMG, Z_IMG, z_ant, z_surf, eps_r,
                        suba_size, up_factor, y_aperture_half=0.02):
    """
    Para cada Y_IMG, selecciona las trazas con |y_ant - y_img| <= y_aperture_half
    y ejecuta SCFBP con esa sub‑apertura en Y.
    """
    n_x = len(x_ant_pos)
    n_y = len(y_ant_pos)
    x_flat_full = np.tile(x_ant_pos, n_y)
    y_flat_full = np.repeat(y_ant_pos, n_x)
    traces_flat_full = data_3d_filt.reshape(data_3d_filt.shape[0], -1)

    nx_img = len(X_IMG)
    nz_img = len(Z_IMG)
    ny_img = len(Y_IMG)
    volumen = np.zeros((nx_img, nz_img, ny_img), dtype=np.float64)

    for j, yp in enumerate(Y_IMG):
        mask_y = np.abs(y_flat_full - yp) <= y_aperture_half
        if not np.any(mask_y):
            idx_closest = np.argmin(np.abs(y_flat_full - yp))
            mask_y[idx_closest] = True

        x_sub = x_flat_full[mask_y]
        y_sub = y_flat_full[mask_y]
        traces_sub = traces_flat_full[:, mask_y]

        print(f"\nProcesando plano Y = {yp:.3f} m ({j+1}/{ny_img}) con {len(x_sub)} trazas")

        img_2d, _, _ = scfbp_2d_3daperture(
            traces_sub, x_sub, y_sub, tiempo,
            X_IMG, Z_IMG, yp, z_ant, z_surf, eps_r,
            suba_size, up_factor, return_complex=False, target_x_grid=X_IMG)
        volumen[:, :, j] = img_2d

    return volumen

# =============================================================================
# 8. VISUALIZACIÓN C‑SCAN
# =============================================================================
def visualizar_cscan(volumen, X_IMG, Z_IMG, Y_IMG, z_slice=0.15):
    iz = np.argmin(np.abs(Z_IMG - z_slice))
    slice_xz = volumen[:, :, Y_IMG.size//2]
    slice_xy = volumen[:, iz, :]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(slice_xz.T, aspect='auto', extent=[X_IMG[0], X_IMG[-1], Z_IMG[0], Z_IMG[-1]],
               origin='lower', cmap='gray')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title(f'Corte X‑Z en Y = {Y_IMG[Y_IMG.size//2]:.2f} m')
    plt.colorbar(label='Amplitud')

    plt.subplot(1, 2, 2)
    plt.imshow(slice_xy.T, aspect='auto', extent=[X_IMG[0], X_IMG[-1], Y_IMG[0], Y_IMG[-1]],
               origin='lower', cmap='gray')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'C‑scan a profundidad Z = {z_slice:.2f} m')
    plt.colorbar(label='Amplitud')
    plt.tight_layout()
    plt.show()

    np.save('volumen_scfbp_3d.npy', volumen)
    print("Volumen guardado como 'volumen_scfbp_3d.npy'")

# =============================================================================
# 9. SCRIPT PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    print("=== GENERANDO C‑SCAN (simulaciones 3D) ===")
    x_ant_pos, y_ant_pos, archivos_out = generar_cscan()

    print("=== CARGANDO DATOS 3D ===")
    tiempo, dt, data_3d = cargar_datos_3d(archivos_out)

    print("=== APLICANDO FILTROS Y ENVOLVENTE ===")
    data_filtrada_3d = aplicar_filtros_3d(data_3d, dt)

    # Para vacío, la permitividad es conocida (1.0)
    eps_usar = EPS_R
    print(f"\n=== PROCESANDO VOLUMEN 3D COMPLETO (ε_r = {eps_usar}) ===")
    volumen_3d = procesar_volumen_3d(
        data_filtrada_3d, x_ant_pos, y_ant_pos, tiempo,
        X_IMG, Y_IMG, Z_IMG, Z_ANT, Z_SURFACE, eps_usar,
        SUBA_PER_SIZE, UPSAMPLE_FACTOR, y_aperture_half=Y_APERTURE_HALF)

    print("\n=== VISUALIZANDO RESULTADOS ===")
    visualizar_cscan(volumen_3d, X_IMG, Z_IMG, Y_IMG, z_slice=0.15)

    print("¡Procesamiento SCFBP 3D completado!")