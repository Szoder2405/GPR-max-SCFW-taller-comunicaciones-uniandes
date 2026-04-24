#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCFBP completo para B‑scan con refracción aire‑arena.
Basado en el artículo "Fast SCFBP Algorithm for GPR‑SAR Imaging" (Zhou et al., 2025).
Implementación corregida: FA2 funcional y visualización de profundidad correcta.
"""

import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.fft import fft, ifft, fftfreq, fftshift
from scipy.signal import savgol_filter, hilbert
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import subprocess
import os
import glob
from matplotlib.animation import FFMpegWriter

# =============================================================================
# PARÁMETROS GLOBALES (ajustar según la simulación)
# =============================================================================
BASE_DIR = r"C:\Users\santi\gprmax\gpr_code"
PLANTILLA = "pec_cscan_single.in"          # archivo plantilla con {x_ant} y {y_ant}
OUT_DIR = "temp_bs_output_scfbp"           # carpeta para archivos temporales

# Geometría del escaneo (B‑scan, y_ant fijo)
Y_ANT = 0.30                # coordenada Y de la antena (centro del cubo en Y)
X_START = 0.15              # posición inicial en X (m)
RESOLUCION_X = 0.008
N_TRAZAS = math.ceil((0.45-0.15)/0.008)    # número de trazas (impar para centrar el cubo)
X_END = N_TRAZAS*RESOLUCION_X +  X_START                # posición final en X (m)

# Parámetros físicos
c = 299792458               # velocidad de la luz en vacío (m/s)
Z_ANT = 0.35                # altura de la antena (m)
EPS_R = 4.0                 # permitividad relativa de la arena
v_soil = c / np.sqrt(EPS_R) # velocidad en la arena (m/s)
Z_SURFACE = 0.30             # posición de la superficie (z=30)

# Malla de la imagen (coordenadas absolutas)
X_IMG = np.linspace(0.10, 0.50, 200)   # rango en X (m)
Z_IMG = np.linspace(0.00, 0.35, 150)   # rango en Z (altura, m)
dz = Z_IMG[1] - Z_IMG[0]               # paso en profundidad

# Filtros de pre‑procesamiento
VENTANA_DEWOW = 11
VENTANA_TIME_ZERO = 30
BANDA_PASA = (0.5e9, 4e9)     # rango de frecuencias útil (Hz)

# Parámetros SCFBP
SUBA_PER_SIZE = 8            # número de trazas por sub‑apertura inicial
UPSAMPLE_FACTOR = 2          # factor de upsampling en cada fusión
OUT_VIDEO = "scfbp_evolution.mp4"
FPS_VIDEO = 2

# =============================================================================
# 1. FUNCIONES AUXILIARES: refracción y tiempo de viaje
# =============================================================================
def punto_refraccion(x_ant, x_img, z_img, z_ant, z_surf, eps_r):
    """
    Encuentra la coordenada x_r en la superficie (z=z_surf) que satisface la ley de Snell.
    Resuelve: (x_r - x_ant)/R1 = (x_r - x_img)/R2 * sqrt(eps_r)
    donde R1 = sqrt((x_r-x_ant)^2 + (z_ant-z_surf)^2)
          R2 = sqrt((x_r-x_img)^2 + (z_surf-z_img)^2)
    Usa búsqueda de raíz con brentq.
    """
    def f(xr):
        R1 = np.hypot(xr - x_ant, z_ant - z_surf)
        R2 = np.hypot(x_img - xr, z_surf - z_img)   # ← ¡cambio clave!
        if R1 == 0 or R2 == 0:
            return 0.0
        return (xr - x_ant) / R1 - np.sqrt(eps_r) * (x_img - xr) / R2

    # Intervalo de búsqueda: entre las dos coordenadas x
    x_min = min(x_ant, x_img) - 0.05
    x_max = max(x_ant, x_img) + 0.05
    try:
        xr = brentq(f, x_min, x_max, xtol=1e-6)
    except ValueError:
        # Si no hay cambio de signo, usar el punto medio
        xr = (x_ant + x_img) / 2
    return xr

def tiempo_viaje(x_ant, x_img, z_img, z_ant, z_surf, eps_r):
    """Tiempo de ida y vuelta (TWTT) en segundos, considerando refracción."""
    xr = punto_refraccion(x_ant, x_img, z_img, z_ant, z_surf, eps_r)
    R1 = np.hypot(xr - x_ant, z_ant - z_surf)
    R2 = np.hypot(xr - x_img, z_surf - z_img)
    t = 2 * (R1 / c + R2 / (c / np.sqrt(eps_r)))
    return t


def punto_refraccion_aprox(x_ant, x_img, z_img, z_ant, z_surf, eps_r):
    # Ecuación (3) del artículo
    y_c = z_img                     # profundidad del punto (m)
    y_k = z_ant - z_surf            # altura de la antena sobre la superficie (m)
    # Evitar división por cero si y_c == y_k (no ocurre porque y_c <= 0.30, y_k=0.05)
    factor = y_c / (y_c - y_k)
    xr = x_img + (1.0 / np.sqrt(eps_r)) * factor * (x_ant - x_img)
    return xr

def tiempo_viaje_aprox(x_ant, x_img, z_img, z_ant, z_surf, eps_r):
    print(f"Argumentos: x_ant={x_ant}, x_img={x_img}, z_img={z_img}, z_ant={z_ant}, z_surf={z_surf}, eps_r={eps_r}")
    xr = punto_refraccion_aprox(x_ant, x_img, z_img, z_ant, z_surf, eps_r)
    print(f"xr = {xr}")
    R1 = np.hypot(xr - x_ant, z_ant - z_surf)
    R2 = np.hypot(x_img - xr, z_surf - z_img)
    print(f"R1 = {R1} m, R2 = {R2} m")
    t = 2.0 * (R1 / c + R2 / (c / np.sqrt(eps_r)))
    print(f"t_seg = {t} s")
    return t

# =============================================================================
# 2. GENERACIÓN DE DATOS (simulaciones gprMax)
# =============================================================================
def generar_bscan():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    x_positions = np.linspace(X_START, X_END, N_TRAZAS)
    archivos_out = []

    with open(os.path.join(BASE_DIR, PLANTILLA), 'r') as f:
        plantilla = f.read()

    for i, x_ant in enumerate(x_positions):
        contenido = plantilla.replace('{x_ant}', str(x_ant)).replace('{y_ant}', str(Y_ANT))
        in_file = os.path.join(OUT_DIR, f"temp_{i:03d}.in")
        with open(in_file, 'w') as f:
            f.write(contenido)

        out_file = in_file.replace('.in', '.out')
        if not os.path.exists(out_file):
            print(f"Ejecutando traza {i+1}/{N_TRAZAS} en x={x_ant:.3f}, y={Y_ANT:.3f} ...")
            subprocess.run(f"python -m gprMax {in_file}", shell=True, check=True)
        else:
            print(f"Traza {i+1} ya existe, omitiendo simulación.")
        archivos_out.append(out_file)

    return x_positions, archivos_out

# =============================================================================
# 3. CARGA Y FILTRADO DE DATOS
# =============================================================================
def cargar_datos(archivos_out):
    with h5py.File(archivos_out[0], 'r') as f:
        dt = f.attrs['dt']
        nt = f.attrs['Iterations']
        tiempo = np.arange(nt) * dt * 1e9   # nanosegundos

    n_trazas = len(archivos_out)
    data = np.zeros((nt, n_trazas), dtype=np.complex128)

    for i, out_file in enumerate(archivos_out):
        with h5py.File(out_file, 'r') as f:
            Vinc = f['tls']['tl1']['Vinc'][:]
            Vtotal = f['tls']['tl1']['Vtotal'][:]
            Vref = Vtotal - Vinc
            data[:, i] = Vref   # señal compleja original (sin Hilbert)
    return tiempo, dt, data

def aplicar_filtros(data, dt):
    # Dewow
    if VENTANA_DEWOW > 0 and VENTANA_DEWOW % 2 == 1:
        data = data - savgol_filter(data, VENTANA_DEWOW, 1, axis=0)

    # Time-zero
    n_samples, n_traces = data.shape
    data_tz = np.zeros_like(data)
    for i in range(n_traces):
        ventana = data[:VENTANA_TIME_ZERO, i]
        idx_peak = np.argmax(np.abs(ventana))
        data_tz[:, i] = np.roll(data[:, i], -idx_peak)

    # Background removal
    data_bg = data_tz - np.mean(data_tz, axis=1, keepdims=True)

    # Band-pass
    freq = fftfreq(data_bg.shape[0], d=dt)
    mask = (np.abs(freq) >= BANDA_PASA[0]) & (np.abs(freq) <= BANDA_PASA[1])
    data_f = fft(data_bg, axis=0)
    data_f[~mask] = 0
    data_filtrada = np.real(ifft(data_f, axis=0))

    # Envolvente de Hilbert (señal real positiva)
    data_env = np.abs(hilbert(data_filtrada, axis=0))
    return data_env

# =============================================================================
# 4. BACK-PROJECTION PARA UNA SUBA PERTURA (dominio tiempo, con refracción)
# =============================================================================
def sub_image_bp(traces, x_ant, tiempo, x_grid, z_grid, z_ant, z_surf, eps_r):
    """
    traces: matriz (nt, n_traces_sub) – señales ya filtradas y envolvente.
    x_ant: lista de posiciones de antena para esta subapertura.
    Retorna imagen (len(x_grid), len(z_grid)) con amplitudes.
    """
    nx, nz = len(x_grid), len(z_grid)
    img = np.zeros((nx, nz))
    n_sub = len(x_ant)

    # Pre‑calcular interpoladores para cada traza
    interp = [interp1d(tiempo, traces[:, i], kind='linear', bounds_error=False, fill_value=0) for i in range(n_sub)]

    for ix, xp in enumerate(x_grid):
        for iz, zp in enumerate(z_grid):
            suma = 0.0
            for i in range(n_sub):
                t = tiempo_viaje(x_ant[i], xp, zp, z_ant, z_surf, eps_r) * 1e9   # a ns
                if 0 < t < tiempo[-1]:
                    suma += interp[i](t)
            img[ix, iz] = suma
    return img

def bp_directo(traces, x_ant, tiempo, x_grid, z_grid, z_ant, z_surf, eps_r):
    nx, nz = len(x_grid), len(z_grid)
    img = np.zeros((nx, nz))
    interp = [interp1d(tiempo, traces[:, i], kind='linear', bounds_error=False, fill_value=0) for i in range(len(x_ant))]
    for ix, xp in enumerate(x_grid):
        for iz, zp in enumerate(z_grid):
            suma = 0.0
            for i in range(len(x_ant)):
                t = tiempo_viaje(x_ant[i], xp, zp, z_ant, z_surf, eps_r) * 1e9  # a ns
                if 0 < t < tiempo[-1]:
                    suma += interp[i](t)
            img[ix, iz] = suma
    return img

# =============================================================================
# 5. COMPRESIÓN ESPECTRAL (FA1 y FA2) según el artículo
# =============================================================================
def aplicar_FA1(img, x_grid, z_grid, z_ant, z_surf, eps_r, Kc):
    """
    FA1: Alineación del centro del espectro en el dominio de la imagen.
    Ecuación (27) del artículo.
    """
    nx, nz = len(x_grid), len(z_grid)
    eps_prime = (1/np.sqrt(eps_r) - 1)**2
    y_k = z_ant - z_surf   # altura de la antena sobre la superficie
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
    """
    FA2: Eliminación de la inclinación del espectro en el dominio de la frecuencia de rango.
    img_freq_range: array (nx, nz) en el dominio (x, Ky) con Ky = frecuencia de rango.
    dK_vec: vector (nz,) con ΔK = K - Kc para cada columna de frecuencia.
    """
    nx, nz = img_freq_range.shape
    eps_prime = (1/np.sqrt(eps_r) - 1)**2
    y_k = z_ant - z_surf
    FA2 = np.zeros((nx, nz), dtype=np.complex128)
    for ix, x in enumerate(x_grid):
        for iz in range(nz):
            z = z_grid[iz]   # altura real
            dK = dK_vec[iz]
            term1 = -dK * np.sqrt(eps_prime * x**2 + y_k**2)
            term2 = -dK * np.sqrt(eps_r) * np.sqrt(x**2 / eps_r + z**2)
            phase = term1 + term2
            FA2[ix, iz] = np.exp(1j * phase)
    return img_freq_range * FA2

# =============================================================================
# FUNCIONES NUEVAS PARA INVERSIÓN DE PERMITIVIDAD (Sección IV del artículo)
# =============================================================================

def estimate_phase_error_from_image(img_complex, x_grid, z_grid, z_depth_interest,
                                    Kc, dK_vec, z_ant, z_surf, eps_guess):
    """
    Estima el error de fase φ_e(Kx) a partir de una imagen compleja obtenida
    con una permitividad errónea. Aplica FA1 y FA2, luego FFT en acimut y
    ajusta un polinomio cuadrático en Kx en la profundidad de interés.

    Parámetros
    ----------
    img_complex : ndarray (nx, nz) complejo
        Imagen obtenida con SCFBP (sin tomar valor absoluto).
    x_grid : ndarray
        Eje X de la imagen.
    z_grid : ndarray
        Eje Z (altura) de la imagen.
    z_depth_interest : float
        Profundidad (coordenada Z) donde se extraerá la fase.
    Kc : float
        Número de onda central.
    dK_vec : ndarray
        Vector ΔK = K - Kc para el eje de rango.
    z_ant, z_surf, eps_guess : floats
        Geometría y permitividad supuesta.

    Retorna
    -------
    p : float
        Coeficiente cuadrático del ajuste φ_e(Kx) ≈ p * Kx^2.
    Kx_vals : ndarray
        Valores de Kx correspondientes a la línea extraída.
    phase_error : ndarray
        Fase estimada (desenvuelta) en función de Kx.
    """
    # 1. Aplicar FA1 y FA2 (compresión espectral) para alinear los espectros
    img_fa1 = aplicar_FA1(img_complex, x_grid, z_grid, z_ant, z_surf, eps_guess, Kc)
    fft_range = fft(img_fa1, axis=1)                     # FFT en rango (eje Z)
    fft_range = aplicar_FA2(fft_range, x_grid, z_grid, Kc, dK_vec,
                            z_ant, z_surf, eps_guess)

    # 2. FFT en acimut para pasar al dominio (Kx, z)
    fft_az = fft(fft_range, axis=0)                      # (nx, nz)
    nx, nz = fft_az.shape
    dx = x_grid[1] - x_grid[0]
    Kx = 2 * np.pi * fftfreq(nx, d=dx)                   # número de onda en acimut

    # 3. Seleccionar la profundidad de interés
    iz = np.argmin(np.abs(z_grid - z_depth_interest))
    signal_Kx = fft_az[:, iz]                            # señal compleja en esa profundidad

    # 4. Aislar la región con energía significativa (evitar ruido)
    mask = np.abs(signal_Kx) > 0.1 * np.max(np.abs(signal_Kx))
    if not np.any(mask):
        mask = np.ones_like(signal_Kx, dtype=bool)
    Kx_masked = Kx[mask]
    phase = np.angle(signal_Kx[mask])
    phase_unwrapped = np.unwrap(phase)

    # 5. Ajuste polinómico de grado 2: φ(Kx) = p2 * Kx^2 + p1 * Kx + p0
    coeffs = np.polyfit(Kx_masked, phase_unwrapped, 2)
    p2, p1, p0 = coeffs

    # 6. Visualización opcional (comentada para no saturar)
    # plt.figure()
    # plt.plot(Kx_masked, phase_unwrapped, 'b.', label='Fase estimada')
    # Kx_fit = np.linspace(Kx_masked.min(), Kx_masked.max(), 200)
    # plt.plot(Kx_fit, np.polyval(coeffs, Kx_fit), 'r-', label='Ajuste cuadrático')
    # plt.xlabel('Kx (rad/m)')
    # plt.ylabel('Fase (rad)')
    # plt.title('Error de fase estimado')
    # plt.legend()
    # plt.show()

    return p2, Kx_masked, phase_unwrapped


def invert_permittivity_from_p(p, Kc, x_ref=0.30, z_ref=0.15,
                               z_ant=0.35, z_surf=0.30, eps_guess=2.0):
    """
    Invierte ε_r a partir del coeficiente cuadrático p usando la relación
    deducida en el artículo (ecs. 38-43). Se emplea un método de bisección.

    Parámetros
    ----------
    p : float
        Coeficiente cuadrático de φ_e(Kx) ≈ p * Kx^2.
    Kc : float
        Número de onda central (rad/m).
    x_ref, z_ref : floats
        Coordenadas del punto de referencia utilizado para la estimación
        (normalmente el centro del blanco).
    z_ant, z_surf, eps_guess : floats
        Geometría y permitividad inicial (solo se usa para el intervalo de búsqueda).

    Retorna
    -------
    eps_inv : float
        Permitividad relativa estimada.
    """
    # Funciones auxiliares definidas en el artículo
    def gamma_k(z):
        # altura efectiva de la antena sobre la superficie
        return z_ant - z_surf

    def gamma_c(z):
        # profundidad del punto
        return z_surf - z

    y_k = gamma_k(z_ref)
    y_c = gamma_c(z_ref)

    # Relación teórica p(ε_r) según ec. (43) junto con (41)
    def p_theoretical(eps_r):
        # g2 aproximado (ec. 41)
        term1 = (np.sqrt(eps_guess) - np.sqrt(eps_r)) / (eps_guess * y_c)
        term2 = np.sqrt(eps_r) * (np.sqrt(eps_guess) + 1) * \
                (1/np.sqrt(eps_guess) - 1/np.sqrt(eps_r)) * (2 / y_k)
        g2 = term1 + term2
        denom = Kc * (eps_r / y_k + 1 / (np.sqrt(eps_r) * y_c))**2
        return g2 / denom

    # Función para bisección
    def f(eps_r):
        return p_theoretical(eps_r) - p

    # Intervalo de búsqueda (basado en valores típicos de arena: 3 a 10)
    eps_min, eps_max = 2.5, 12.0
    try:
        eps_inv = brentq(f, eps_min, eps_max, xtol=1e-4)
    except ValueError:
        print("Bisección falló; se devuelve el valor medio del intervalo.")
        eps_inv = (eps_min + eps_max) / 2

    return eps_inv


def pga_residual_correction(img_complex, x_grid, z_grid, Kc, dK_vec,
                            z_ant, z_surf, eps_r):
    """
    Aplica una corrección de fase residual (estilo PGA) a la imagen
    después de la inversión de permitividad. Opera en el dominio Kx.

    Retorna la imagen corregida (compleja).
    """
    # 1. Compresión espectral
    img_fa1 = aplicar_FA1(img_complex, x_grid, z_grid, z_ant, z_surf, eps_r, Kc)
    fft_range = fft(img_fa1, axis=1)
    fft_range = aplicar_FA2(fft_range, x_grid, z_grid, Kc, dK_vec, z_ant, z_surf, eps_r)

    # 2. FFT acimut
    fft_az = fft(fft_range, axis=0)
    nx, nz = fft_az.shape
    dx = x_grid[1] - x_grid[0]
    Kx = 2 * np.pi * fftfreq(nx, d=dx)

    # 3. Estimar fase residual en una profundidad de alto contraste
    #    (por simplicidad tomamos la profundidad con máxima energía promedio)
    energy = np.sum(np.abs(fft_az), axis=0)
    iz = np.argmax(energy)
    signal_Kx = fft_az[:, iz]
    mask = np.abs(signal_Kx) > 0.1 * np.max(np.abs(signal_Kx))
    phase = np.angle(signal_Kx[mask])
    Kx_masked = Kx[mask]
    coeffs = np.polyfit(Kx_masked, np.unwrap(phase), 2)
    phase_correction = np.polyval(coeffs, Kx)

    # 4. Aplicar corrección a todos los rangos
    correction = np.exp(-1j * phase_correction[:, np.newaxis])
    fft_az_corrected = fft_az * correction

    # 5. Transformada inversa
    img_corrected_range = ifft(fft_az_corrected, axis=0)
    img_corrected = ifft(img_corrected_range, axis=1)

    return img_corrected


# =============================================================================
# MODIFICACIÓN DE scfbp_2d_completo PARA DEVOLVER IMAGEN COMPLEJA
# =============================================================================

# =============================================================================
# 6. FUSIÓN JERÁRQUICA COMPLETA (SCFBP)
# =============================================================================
def scfbp_2d_completo(traces, x_ant, tiempo, x_grid_base, z_grid, z_ant, z_surf, eps_r,
                      suba_size, up_factor, return_complex=False):
    """
    Versión modificada: si return_complex=True, devuelve la imagen compleja
    final sin tomar valor absoluto. Los frames del vídeo siempre se toman en magnitud.

    Implementa el SCFBP completo:
    1. Divide en subaperturas.
    2. Genera sub‑imágenes con BP (tiempo).
    3. Aplica FA1, FFT rango, FA2, FFT azimuth, upsampling, IFFT, suma.
    4. Recursión.
    """
    n_ant = len(x_ant)
    # Crear subaperturas
    suba_indices = [list(range(i, min(i+suba_size, n_ant))) for i in range(0, n_ant, suba_size)]
    sub_images = []
    sub_grids_x = []

    print("Generando sub‑imágenes iniciales...")
    for idxs in suba_indices:
        x_sub = x_ant[idxs]
        traces_sub = traces[:, idxs]
        img = sub_image_bp(traces_sub, x_sub, tiempo, x_grid_base, z_grid, z_ant, z_surf, eps_r)
        sub_images.append(img)
        sub_grids_x.append(x_grid_base.copy())

    # Número de onda central (aproximado a partir del pulso)
    fc = (BANDA_PASA[0] + BANDA_PASA[1]) / 2  # frecuencia central (Hz)
    Kc = 4 * np.pi * fc / c   # número de onda central (rad/m)

    # Preparar vector dK para el eje de rango (se calcula una sola vez)
    # Relación: frecuencia espacial ky -> número de onda K = 2 * ky / sqrt(eps_r)
    # (ver explicación en la documentación)
    ky = fftfreq(len(z_grid), d=dz)               # rad/m
    K = ky / np.sqrt(eps_r)                   # número de onda asociado
    dK_vec = K - Kc

    frames = []  # para vídeo

    nivel = 0
    while len(sub_images) > 1:
        print(f"Fusión nivel {nivel+1}, {len(sub_images)} sub‑imágenes")
        new_images = []
        new_grids_x = []
        for k in range(0, len(sub_images), 2):
            if k+1 < len(sub_images):
                imgA, gridA = sub_images[k], sub_grids_x[k]
                imgB, gridB = sub_images[k+1], sub_grids_x[k+1]

                # 1. FA1 (dominio imagen)
                imgA = aplicar_FA1(imgA, gridA, z_grid, z_ant, z_surf, eps_r, Kc)
                imgB = aplicar_FA1(imgB, gridB, z_grid, z_ant, z_surf, eps_r, Kc)

                # 2. FFT en rango (eje Z)
                fftA = fft(imgA, axis=1)
                fftB = fft(imgB, axis=1)

                # 3. FA2 (dominio frecuencia de rango)
                fftA = aplicar_FA2(fftA, gridA, z_grid, Kc, dK_vec, z_ant, z_surf, eps_r)
                fftB = aplicar_FA2(fftB, gridB, z_grid, Kc, dK_vec, z_ant, z_surf, eps_r)

                # 4. FFT en azimuth (eje X) para cada frecuencia de rango
                fftA_az = fft(fftA, axis=0)
                fftB_az = fft(fftB, axis=0)

                # 5. Upsampling en azimuth (rellenar con ceros en el dominio de la frecuencia)
                nxA, nz = fftA_az.shape
                nxB = fftB_az.shape[0]
                new_nx = int(max(nxA, nxB) * up_factor)
                fftA_up = np.zeros((new_nx, nz), dtype=np.complex128)
                fftB_up = np.zeros((new_nx, nz), dtype=np.complex128)
                startA = (new_nx - nxA) // 2
                fftA_up[startA:startA+nxA, :] = fftA_az
                startB = (new_nx - nxB) // 2
                fftB_up[startB:startB+nxB, :] = fftB_az

                # 6. Sumar coherentemente
                fft_fused = fftA_up + fftB_up

                # 7. IFFT en azimuth
                img_fused_az = ifft(fft_fused, axis=0)

                # 8. IFFT en rango (y deshacer FA2 y FA1)
                # Nota: No se deshacen explícitamente porque luego se volverán a aplicar
                # en el siguiente nivel. Para la última fusión, la imagen resultante
                # se toma en magnitud.
                img_fused = ifft(img_fused_az, axis=1)
                # NO tomar magnitud aquí
                # img_fused se mantiene complejo

                # Nueva grilla X (más fina)
                x_min = min(gridA[0], gridB[0])
                x_max = max(gridA[-1], gridB[-1])
                new_grid = np.linspace(x_min, x_max, new_nx)

                new_images.append(img_fused)
                new_grids_x.append(new_grid)
            else:
                new_images.append(sub_images[k])
                new_grids_x.append(sub_grids_x[k])

        # Guardar frame (imagen de la primera sub‑imagen fusionada)
        if new_images:
            frame = new_images[0]
            if np.max(frame) > 0:
                frame /= np.max(frame)
            frames.append(frame)

        sub_images = new_images
        sub_grids_x = new_grids_x
        nivel += 1

    # Al final, antes de devolver:
    final_img_complex = sub_images[0]          # imagen compleja final
    if return_complex:
        return final_img_complex, frames, sub_grids_x[0]
    else:
        final_img_mag = np.abs(final_img_complex)
        if np.max(final_img_mag) > 0:
            final_img_mag /= np.max(final_img_mag)
        return final_img_mag, frames, sub_grids_x[0]

# =============================================================================
# 7. VISUALIZACIÓN (profundidad aumentando hacia abajo)
# =============================================================================

def graficar_resultados(final_img, x_grid, z_grid, z_ant, frames, x_final, output_video, guardar_video=False):
    """
    Muestra dos visualizaciones:
    1) Imagen purista (sin transformaciones) con coordenadas originales.
    2) Imagen con las mismas coordenadas absolutas, más regiones semitransparentes de aire, arena y bloque PEC.
    Opcionalmente genera un vídeo de evolución con la misma corrección.
    """
    # Definir coordenadas de la geometría (en metros, coordenadas absolutas Z)
    Z_SUPERFICIE = 0.30      # superficie de la arena
    Z_ANTENA = z_ant         # 0.35 m
    X_BLOQUE_MIN, X_BLOQUE_MAX = 0.25, 0.35
    Z_BLOQUE_MIN, Z_BLOQUE_MAX = 0.10, 0.20

    # =========================================================================
    # 1. VISUALIZACIÓN PURISTA (sin añadidos)
    # =========================================================================
    final_magnitud = np.abs(final_img)
    plt.figure(figsize=(10, 6))
    plt.imshow(final_magnitud.T, aspect='auto',
               extent=[x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]],
               origin='lower', cmap='gray')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m) - altura absoluta (crece hacia arriba)')
    plt.title('SCFBP - Imagen original (sin transformaciones)')
    plt.colorbar(label='Amplitud normalizada')
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 2. VISUALIZACIÓN CON REGIONES (coordenadas absolutas)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # Mostrar la imagen de amplitud
    im = ax2.imshow(final_magnitud.T, aspect='auto',
                    extent=[x_grid[0], x_grid[-1], z_grid[0], z_grid[-1]],
                    origin='lower', cmap='gray')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m) - altura absoluta (crece hacia arriba)')
    ax2.set_title('SCFBP - Regiones del medio (coordenadas absolutas)')
    cbar = plt.colorbar(im, ax=ax2, label='Amplitud normalizada')

    # Región de aire (desde la superficie hasta la antena, y por encima si la imagen lo permite)
    # La imagen cubre hasta z_grid[-1] (0.30), pero el aire está por encima de 0.30 hasta 0.35
    # Dibujamos un rectángulo que vaya desde Z_SUPERFICIE hasta Z_ANTENA
    altura_aire = Z_ANTENA - Z_SUPERFICIE  # 0.05 m
    ax2.add_patch(plt.Rectangle((x_grid[0], Z_SUPERFICIE), x_grid[-1]-x_grid[0], altura_aire,
                                facecolor='cyan', alpha=0.2, edgecolor='none'))

    # Región de arena (desde el fondo del dominio hasta la superficie)
    # El fondo de la imagen es z_grid[0] (0.05), pero la arena continúa hasta z=0
    # Para simplificar, dibujamos desde z_grid[0] hasta Z_SUPERFICIE
    ax2.add_patch(plt.Rectangle((x_grid[0], z_grid[0]), x_grid[-1]-x_grid[0], Z_SUPERFICIE - z_grid[0],
                                facecolor='peru', alpha=0.2, edgecolor='none'))

    # Bloque metálico (PEC)
    ax2.add_patch(plt.Rectangle((X_BLOQUE_MIN, Z_BLOQUE_MIN),
                                 X_BLOQUE_MAX - X_BLOQUE_MIN,
                                 Z_BLOQUE_MAX - Z_BLOQUE_MIN,
                                 facecolor='red', alpha=0.3, edgecolor='yellow', linewidth=1.5))

    # Línea horizontal para la superficie (z=0.30)
    ax2.axhline(y=Z_SUPERFICIE, color='blue', linestyle='--', linewidth=1, label='Superficie arena')
    # Línea para la antena (z=0.35)
    ax2.axhline(y=Z_ANTENA, color='green', linestyle='--', linewidth=1, label='Antena')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 3. VÍDEO DE EVOLUCIÓN (coordenadas absolutas)
    # =========================================================================
    if guardar_video and frames:
        print("Generando vídeo... (puede tomar unos segundos)")
        fig_vid, ax_vid = plt.subplots(figsize=(8, 6))
        # Usar el primer frame para inicializar
        first_frame_mag = np.abs(frames[0])
        im_vid = ax_vid.imshow(first_frame_mag.T, aspect='auto',
                               extent=[x_final[0], x_final[-1], z_grid[0], z_grid[-1]],
                               origin='lower', cmap='gray', vmin=0, vmax=1)
        ax_vid.set_xlabel('X (m)')
        ax_vid.set_ylabel('Z (m) - altura absoluta')
        ax_vid.set_title('Evolución SCFBP')
        # Añadir las mismas regiones y líneas
        ax_vid.add_patch(plt.Rectangle((x_final[0], Z_SUPERFICIE), x_final[-1]-x_final[0], altura_aire,
                                       facecolor='cyan', alpha=0.2, edgecolor='none'))
        ax_vid.add_patch(plt.Rectangle((x_final[0], z_grid[0]), x_final[-1]-x_final[0], Z_SUPERFICIE - z_grid[0],
                                       facecolor='peru', alpha=0.2, edgecolor='none'))
        ax_vid.add_patch(plt.Rectangle((X_BLOQUE_MIN, Z_BLOQUE_MIN),
                                       X_BLOQUE_MAX - X_BLOQUE_MIN,
                                       Z_BLOQUE_MAX - Z_BLOQUE_MIN,
                                       facecolor='red', alpha=0.3, edgecolor='yellow', linewidth=1.5))
        ax_vid.axhline(y=Z_SUPERFICIE, color='blue', linestyle='--', linewidth=1)
        ax_vid.axhline(y=Z_ANTENA, color='green', linestyle='--', linewidth=1)
        plt.colorbar(im_vid, ax=ax_vid, label='Amplitud')
        
        writer = FFMpegWriter(fps=FPS_VIDEO, bitrate=1800)
        writer.setup(fig_vid, output_video, dpi=100)
        for img in frames:
            img_mag = np.abs(img)
            im_vid.set_array(img_mag.T)
            writer.grab_frame()
        writer.finish()
        plt.close(fig_vid)
        print(f"Vídeo guardado como {output_video}")
    else:
        print("Generación de vídeo desactivada (guardar_video=False).")

# =============================================================================
# 8. SCRIPT PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    """LO QUE EL CÓDIGO NO IMPLEMENTA (PARTE DE INVERSIÓN DE PERMITIVIDAD)
    ------------------------------------------------------------------

    El artículo dedica la Sección IV a la inversión de la permitividad y 
    compensación de errores de fase. Allí se describe:

    1. Estimación del error de fase:
    φ_e(Kx, Ky) a partir de la imagen desenfocada.

    2. Relación cuadrática:
    Entre el error de fase y la permitividad (ecuaciones 38-43).

    3. Métodos de inversión:
    Uso de algoritmos como la dicotomía para invertir ε_r.

    4. Re-procesamiento:
    Ejecución del SCFBP con la permitividad actualizada y 
    compensación residual (PGA).

    NOTA: 
    Tu código actual no incluye ninguna de estas etapas. Asume que la 
    permitividad es conocida y correcta (EPS_R = 4.0). No realiza 
    estimación de error de fase, ni actualización de ε_r, ni 
    compensación adicional."""

    print("=== GENERANDO B‑SCAN (simulaciones) ===")
    x_ant_positions, archivos_out = generar_bscan()

    print("=== CARGANDO DATOS ===")
    tiempo, dt, data_compleja = cargar_datos(archivos_out)

    print("=== APLICANDO FILTROS Y ENVOLVENTE ===")
    data_filtrada = aplicar_filtros(data_compleja, dt)   # señal real positiva

    

    # Selecciona la traza central (donde la antena está sobre el bloque, x≈0.30)
    idx_centro = np.argmin(np.abs(x_ant_positions - 0.30))
    traza_central = data_filtrada[:, idx_centro]

    """
    plt.figure(figsize=(10,4))
    plt.plot(tiempo, traza_central)
    plt.xlabel('Tiempo (ns)')
    plt.ylabel('Amplitud')
    plt.title('A‑scan en x = 0.30 m')
    plt.grid(True)
    plt.show()

    """

    # Calcula el TWTT teórico para el bloque (z=0.15) justo debajo de la antena
    x_ant = 0.30
    x_img = 0.30
    z_img = 0.15
    t_teorico = tiempo_viaje(x_ant, x_img, z_img, Z_ANT, Z_SURFACE, EPS_R) * 1e9
    print(f"TWTT teórico para el bloque (modelo exacto): {t_teorico:.2f} ns")

    # Busca el pico máximo en la traza alrededor de ese tiempo
    ventana = (t_teorico - 0.5, t_teorico + 0.5)
    idx_ventana = np.where((tiempo >= ventana[0]) & (tiempo <= ventana[1]))[0]
    if len(idx_ventana) > 0:
        pico = np.max(np.abs(traza_central[idx_ventana]))
        print(f"Máxima amplitud en ventana {ventana} ns: {pico:.3e}")
    else:
        print("No hay datos en esa ventana temporal.")

    # -------------------------------------------------------------------------
    # PRIMERA PASADA: con permitividad errónea (estimación)
    # -------------------------------------------------------------------------

    eps_guess = EPS_R                     # valor inicial incorrecto
    print(f"\n=== PRIMERA PASADA SCFBP (ε_r = {eps_guess}) ===")
    img_complex, frames, x_final = scfbp_2d_completo(
        data_filtrada, x_ant_positions, tiempo,
        X_IMG, Z_IMG, Z_ANT, Z_SURFACE, eps_guess,
        SUBA_PER_SIZE, UPSAMPLE_FACTOR,
        return_complex=True)            # <-- obtener imagen compleja

    # Parámetros para la estimación
    fc = (BANDA_PASA[0] + BANDA_PASA[1]) / 2
    Kc = 4 * np.pi * fc / c
    ky = fftfreq(len(Z_IMG), d=dz)
    K = ky / np.sqrt(eps_guess)         # usar eps_guess para definir K
    dK_vec = K - Kc

    # Profundidad de interés: centro del bloque metálico (z = 0.15 m)
    z_ref = 0.15
    print("Estimando error de fase...")
    p, Kx_vals, phase_err = estimate_phase_error_from_image(
        img_complex, x_final, Z_IMG, z_ref,
        Kc, dK_vec, Z_ANT, Z_SURFACE, eps_guess)

    print(f"Coeficiente cuadrático estimado p = {p:.3e}")

    # Inversión de permitividad
    eps_inverted = invert_permittivity_from_p(
        p, Kc, x_ref=0.30, z_ref=z_ref,
        z_ant=Z_ANT, z_surf=Z_SURFACE, eps_guess=eps_guess)

    print(f"Permitividad invertida: ε_r = {eps_inverted:.3f}")

    # -------------------------------------------------------------------------
    # SEGUNDA PASADA: con permitividad corregida
    # -------------------------------------------------------------------------
    print(f"\n=== SEGUNDA PASADA SCFBP (ε_r = {eps_inverted:.3f}) ===")
    final_img_corrected, frames2, x_final2 = scfbp_2d_completo(
        data_filtrada, x_ant_positions, tiempo,
        X_IMG, Z_IMG, Z_ANT, Z_SURFACE, eps_inverted,
        SUBA_PER_SIZE, UPSAMPLE_FACTOR,
        return_complex=False)            # ya devuelve magnitud normalizada

    # -------------------------------------------------------------------------
    # COMPENSACIÓN RESIDUAL (OPCIONAL)
    # -------------------------------------------------------------------------
    # Si se desea aplicar PGA residual, se puede obtener la imagen compleja
    # de la segunda pasada y corregir:
    img_complex2, _, _ = scfbp_2d_completo(
        data_filtrada, x_ant_positions, tiempo,
        X_IMG, Z_IMG, Z_ANT, Z_SURFACE, eps_inverted,
        SUBA_PER_SIZE, UPSAMPLE_FACTOR,
        return_complex=False)
    img_final = pga_residual_correction(img_complex2, x_final2, Z_IMG,
                                         Kc, dK_vec, Z_ANT, Z_SURFACE, eps_inverted)
    final_img_corrected = np.abs(img_final) / np.max(np.abs(img_final))

    # -------------------------------------------------------------------------
    # VISUALIZACIÓN
    # -------------------------------------------------------------------------
    print("=== MOSTRANDO RESULTADOS ===")
    graficar_resultados(final_img_corrected, x_final2, Z_IMG, Z_ANT,
                        frames2, x_final2, OUT_VIDEO, guardar_video=False)

    print("\nResumen de permitividad:")
    print(f"  Valor real:        {EPS_R}")
    print(f"  Valor inicial:     {eps_guess}")
    print(f"  Valor invertido:   {eps_inverted:.3f}")

    """
    print("=== EJECUTANDO BP DIRECTO ===")
    img_bp = bp_directo(data_filtrada, x_ant_positions, tiempo, X_IMG, Z_IMG, Z_ANT, Z_SURFACE, EPS_R)
    img_bp = np.abs(img_bp)
    if np.max(img_bp) > 0:
        img_bp /= np.max(img_bp)

    plt.figure(figsize=(10, 6))
    plt.imshow(img_bp.T, aspect='auto',
            extent=[X_IMG[0], X_IMG[-1], Z_IMG[0], Z_IMG[-1]],
            origin='lower', cmap='gray')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('BP Directo - Todas las trazas')
    plt.colorbar(label='Amplitud normalizada')
    plt.show()

    print("¡Proceso completado!")
    """