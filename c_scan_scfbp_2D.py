#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCFBP extendido a C‑scan 3D (barrido en X e Y)
Basado en el artículo "Fast SCFBP Algorithm for GPR‑SAR Imaging" (Zhou et al., 2025).
"""

import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import savgol_filter, hilbert
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import brentq
import plotly.graph_objects as go
import subprocess
import os
import glob

# =============================================================================
# PARÁMETROS GLOBALES (C‑scan)
# =============================================================================
BASE_DIR = r"C:\Users\santi\gprmax\gpr_code"
PLANTILLA = "pec_cscan_single.in"          # plantilla con {x_ant} y {y_ant}
OUT_DIR = "temp_cscan_output"              # carpeta para archivos temporales

# Barrido en X
X_START = 0.15
X_END   = 0.45
RESOLUCION_X = 0.008
N_TRAZAS_X = 30 #math.ceil((X_END - X_START) / RESOLUCION_X)

# Barrido en Y (nuevo)
Y_START = 0.15
Y_END   = 0.45
RESOLUCION_Y = 0.008
N_TRAZAS_Y = 30 #math.ceil((Y_END - Y_START) / RESOLUCION_Y)

# Parámetros físicos
c = 299792458
Z_ANT = 0.35
EPS_R = 1.0                 # permitividad real de la arena 4.0
Z_SURFACE = 0.0 # 0.30

# Malla de la imagen 3D
X_IMG = np.linspace(X_START,X_END, 80)
Z_IMG = np.linspace(0.05, 0.35, 60)
Y_IMG = np.linspace(Y_START, Y_END, 80)   # nuevo eje Y para la imagen 3D
dz = Z_IMG[1] - Z_IMG[0]

# Filtros
VENTANA_DEWOW = 11
VENTANA_TIME_ZERO = 162 #VENTANA_TIME_ZERO = int(2.5e-9 / dt)
#BANDA_PASA = (0.5e9, 4e9)
FRECUENCIA_CENTRAL_PULSO = 1.5e9
BANDA_PASA = (0.5e9, 2.5e9)

# Parámetros SCFBP
SUBA_PER_SIZE = 8
UPSAMPLE_FACTOR_FALLBACK = 2 

# =============================================================================
# 1. FUNCIONES GEOMÉTRICAS (refracción) – igual que antes
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

def tiempo_viaje(x_ant, x_img, z_img, z_ant, z_surf, eps_r):
    """Tiempo de ida y vuelta (TWTT) en segundos."""
    if np.isclose(eps_r, 1.0):
        # Propagación en línea recta (sin refracción)
        R = np.hypot(x_img - x_ant, z_img - z_ant)
        return 2.0 * R / c
    else:
        xr = punto_refraccion(x_ant, x_img, z_img, z_ant, z_surf, eps_r)
        R1 = np.hypot(xr - x_ant, z_ant - z_surf)
        R2 = np.hypot(xr - x_img, z_surf - z_img)
        return 2.0 * (R1 / c + R2 / (c / np.sqrt(eps_r)))

# =============================================================================
# 2. GENERACIÓN DE DATOS (C‑scan: barrido en X e Y)
# =============================================================================
def generar_cscan():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    x_positions = np.linspace(X_START, X_END, N_TRAZAS_X)
    y_positions = np.linspace(Y_START, Y_END, N_TRAZAS_Y)

    with open(os.path.join(BASE_DIR, PLANTILLA), 'r') as f:
        plantilla = f.read()

    # Lista 2D: archivos_out[ny][nx]
    archivos_out = [[None for _ in range(N_TRAZAS_X)] for _ in range(N_TRAZAS_Y)]

    total = N_TRAZAS_X * N_TRAZAS_Y
    contador = 0
    for iy, y_ant in enumerate(y_positions):
        for ix, x_ant in enumerate(x_positions):
            contenido = plantilla.replace('{x_ant}', str(x_ant)).replace('{y_ant}', str(y_ant))
            in_file = os.path.join(OUT_DIR, f"temp_{ix:03d}_{iy:03d}.in")
            out_file = in_file.replace('.in', '.out')

            # Si el archivo .out ya existe, se omite la simulación
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
# 3. CARGA DE DATOS 3D
# =============================================================================
def cargar_datos_3d(archivos_out):
    """Carga todas las trazas y las organiza en un array 3D: (nt, nx, ny)."""
    with h5py.File(archivos_out[0][0], 'r') as f:
        dt = f.attrs['dt']
        nt = f.attrs['Iterations']
        tiempo = np.arange(nt) * dt   #s

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
    return tiempo, dt, nt, data_3d

def aplicar_filtros_3d(data_3d, dt):
    """Aplica los mismos filtros que en 2D, pero a cada traza independientemente."""
    nt, nx, ny = data_3d.shape
    data_filtrada = np.zeros_like(data_3d, dtype=np.float64)

    for iy in range(ny):
        for ix in range(nx):
            traza = data_3d[:, ix, iy].real  # trabajamos con parte real
            # Dewow
            if VENTANA_DEWOW > 0 and VENTANA_DEWOW % 2 == 1:
                traza = traza - savgol_filter(traza, VENTANA_DEWOW, 1)
            # Time-zero
            ventana = traza[:VENTANA_TIME_ZERO]
            idx_peak = np.argmax(np.abs(ventana))
            traza = np.roll(traza, -idx_peak)
            # Band-pass
            freq = fftfreq(nt, d=dt)
            mask = (np.abs(freq) >= BANDA_PASA[0]) & (np.abs(freq) <= BANDA_PASA[1])
            traza_f = fft(traza)
            traza_f[~mask] = 0
            traza = np.real(ifft(traza_f))
            data_filtrada[:, ix, iy] = traza

    # Background removal (por cada plano X‑Z)
    #for iy in range(ny):
    #    data_filtrada[:, :, iy] -= np.mean(data_filtrada[:, :, iy], axis=1, keepdims=True)

    for iy in range(ny):
        # Calcula la traza promedio de todo el perfil X para esta Y
        traza_promedio = np.mean(data_filtrada[:, :, iy], axis=1) 
        # Resta ese promedio a cada traza
        data_filtrada[:, :, iy] -= traza_promedio[:, np.newaxis]

    # Envolvente de Hilbert
    data_env = np.abs(hilbert(data_filtrada, axis=0))
    return data_env

# =============================================================================
# 4. Funciones SCFBP 2D (idénticas a las del B‑scan, sin cambios)
# =============================================================================
# =============================================================================
# 4. BACK-PROJECTION PARA UNA SUBA PERTURA (dominio tiempo, con refracción)
# =============================================================================
# Version antigua

'''def sub_image_bp(traces, x_ant, tiempo, x_grid, z_grid, z_ant, z_surf, eps_r):
 
    nx, nz = len(x_grid), len(z_grid)
    img = np.zeros((nx, nz))
    n_sub = len(x_ant)

    # Pre‑calcular interpoladores para cada traza
    interp = [interp1d(tiempo, traces[:, i], kind='linear', bounds_error=False, fill_value=0) for i in range(n_sub)]

    for ix, xp in enumerate(x_grid):
        for iz, zp in enumerate(z_grid):
            suma = 0.0
            for i in range(n_sub):
                t = tiempo_viaje(x_ant[i], xp, zp, z_ant, z_surf, eps_r) #* 1e9   # a ns
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
                t = tiempo_viaje(x_ant[i], xp, zp, z_ant, z_surf, eps_r)  # en segundos
                if tiempo[0] <= t <= tiempo[-1]:
                    suma += interp[i](t)
            img[ix, iz] = suma
    return img'''

# Versión corregida
'''
def bp_directo(traces, x_ant, tiempo, x_grid, z_grid, z_ant, z_surf, eps_r):
    nx, nz = len(x_grid), len(z_grid)
    img = np.zeros((nx, nz), dtype=np.complex128)
    interp_real = [interp1d(tiempo, np.real(traces[:, i]), kind='linear', bounds_error=False, fill_value=0) for i in range(len(x_ant))]
    interp_imag = [interp1d(tiempo, np.imag(traces[:, i]), kind='linear', bounds_error=False, fill_value=0) for i in range(len(x_ant))]
    for ix, xp in enumerate(x_grid):
        for iz, zp in enumerate(z_grid):
            suma = 0.0 + 0.0j
            for i in range(len(x_ant)):
                t = tiempo_viaje(x_ant[i], xp, zp, z_ant, z_surf, eps_r)
                if tiempo[0] <= t <= tiempo[-1]:
                    suma += interp_real[i](t) + 1j * interp_imag[i](t)
            img[ix, iz] = suma
    return img

def sub_image_bp(traces, x_ant, tiempo, x_grid, z_grid, z_ant, z_surf, eps_r):
    nx, nz = len(x_grid), len(z_grid)
    img = np.zeros((nx, nz), dtype=np.complex128)
    n_sub = len(x_ant)
    interp_real = [interp1d(tiempo, np.real(traces[:, i]), kind='linear', bounds_error=False, fill_value=0) for i in range(n_sub)]
    interp_imag = [interp1d(tiempo, np.imag(traces[:, i]), kind='linear', bounds_error=False, fill_value=0) for i in range(n_sub)]
    for ix, xp in enumerate(x_grid):
        for iz, zp in enumerate(z_grid):
            suma = 0.0 + 0.0j
            for i in range(n_sub):
                t = tiempo_viaje(x_ant[i], xp, zp, z_ant, z_surf, eps_r)
                if tiempo[0] <= t <= tiempo[-1]:
                    suma += interp_real[i](t) + 1j * interp_imag[i](t)
            img[ix, iz] = suma
    return img
'''

# Versión vectorizada

def bp_directo(traces, x_ant, tiempo, x_grid, z_grid, z_ant, z_surf, eps_r):
    """
    Versión vectorizada de BP directo. Asume que 'tiempo' es uniforme (dt constante).
    """
    nx, nz = len(x_grid), len(z_grid)
    n_ant = len(x_ant)
    nt = len(tiempo)
    dt = tiempo[1] - tiempo[0]

    # Precalcular retardos para todas las combinaciones (n_ant, nx, nz)
    delays = np.zeros((n_ant, nx, nz))
    for i, xa in enumerate(x_ant):
        for ix, xp in enumerate(x_grid):
            for iz, zp in enumerate(z_grid):
                delays[i, ix, iz] = tiempo_viaje(xa, xp, zp, z_ant, z_surf, eps_r)

    img = np.zeros((nx, nz), dtype=np.complex128)
    for i in range(n_ant):
        trace = traces[:, i]                      # señal compleja de la antena i
        t_vals = delays[i, :, :]                  # (nx, nz)
        idx_floor = np.floor(t_vals / dt).astype(int)
        idx_ceil = idx_floor + 1
        w_ceil = (t_vals - idx_floor * dt) / dt
        w_floor = 1.0 - w_ceil

        valid = (idx_floor >= 0) & (idx_ceil < nt)
        np.clip(idx_floor, 0, nt-1, out=idx_floor)
        np.clip(idx_ceil, 0, nt-1, out=idx_ceil)

        contrib = w_floor * trace[idx_floor] + w_ceil * trace[idx_ceil]
        contrib[~valid] = 0.0
        img += contrib

    return img


def sub_image_bp(traces, x_ant, tiempo, x_grid, z_grid, z_ant, z_surf, eps_r):
    """
    Versión vectorizada de BP para una subapertura.
    """
    nx, nz = len(x_grid), len(z_grid)
    n_sub = len(x_ant)
    nt = len(tiempo)
    dt = tiempo[1] - tiempo[0]

    # Precalcular retardos (n_sub, nx, nz)
    delays = np.zeros((n_sub, nx, nz))
    for i, xa in enumerate(x_ant):
        for ix, xp in enumerate(x_grid):
            for iz, zp in enumerate(z_grid):
                delays[i, ix, iz] = tiempo_viaje(xa, xp, zp, z_ant, z_surf, eps_r)

    img = np.zeros((nx, nz), dtype=np.complex128)
    for i in range(n_sub):
        trace = traces[:, i]
        t_vals = delays[i, :, :]
        idx_floor = np.floor(t_vals / dt).astype(int)
        idx_ceil = idx_floor + 1
        w_ceil = (t_vals - idx_floor * dt) / dt
        w_floor = 1.0 - w_ceil

        valid = (idx_floor >= 0) & (idx_ceil < nt)
        np.clip(idx_floor, 0, nt-1, out=idx_floor)
        np.clip(idx_ceil, 0, nt-1, out=idx_ceil)

        contrib = w_floor * trace[idx_floor] + w_ceil * trace[idx_ceil]
        contrib[~valid] = 0.0
        img += contrib

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
                      suba_size, up_factor_fallback=2, return_complex=False, target_x_grid=None):
    n_ant = len(x_ant)
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

    # --- Depuración: mostrar las dos primeras subimágenes y sus espectros ---
    '''
    if len(sub_images) >= 2:
        
        from scipy.fft import fft2, fftshift
        plt.figure(figsize=(12, 5))
        plt.subplot(2, 2, 1)
        plt.imshow(np.abs(sub_images[0]).T, aspect='auto', origin='lower')
        plt.title('Subimagen 0 (magnitud)')
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(np.abs(sub_images[1]).T, aspect='auto', origin='lower')
        plt.title('Subimagen 1 (magnitud)')
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(np.log10(np.abs(fftshift(fft2(sub_images[0]))) + 1e-12).T, aspect='auto')
        plt.title('Espectro 2D Sub0')
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.imshow(np.log10(np.abs(fftshift(fft2(sub_images[1]))) + 1e-12).T, aspect='auto')
        plt.title('Espectro 2D Sub1')
        plt.colorbar()
        plt.tight_layout()
        plt.show()'''

    # Número de onda central
    fc = FRECUENCIA_CENTRAL_PULSO
    #fc = (BANDA_PASA[0] + BANDA_PASA[1]) / 2
    Kc = 4 * np.pi * fc / c

    # Cálculo de dK_vec
    dz = z_grid[1] - z_grid[0]
    ky = 2 * np.pi * fftfreq(len(z_grid), d=dz)
    K = compute_K_from_ky(ky, eps_r)
    dK_vec = K - Kc
    #dK_vec = np.zeros_like(ky) # prueba debug desabilitando FA2

    frames = []
    nivel = 0

    while len(sub_images) > 1:
        print(f"Fusión nivel {nivel+1}, {len(sub_images)} sub‑imágenes")
        new_images = []
        new_grids_x = []
        for k in range(0, len(sub_images), 2):
            if k+1 < len(sub_images):
                imgA, gridA = sub_images[k], sub_grids_x[k]
                imgB, gridB = sub_images[k+1], sub_grids_x[k+1]

                # 1. FA1
                imgA = aplicar_ventana_tukey(imgA, alpha=0.3)
                imgB = aplicar_ventana_tukey(imgB, alpha=0.3)

                imgA = aplicar_FA1(imgA, gridA, z_grid, z_ant, z_surf, eps_r, Kc)
                imgB = aplicar_FA1(imgB, gridB, z_grid, z_ant, z_surf, eps_r, Kc)

                # 2. FFT en rango (eje Z)
                fftA = fft(imgA, axis=1)
                fftB = fft(imgB, axis=1)

                # 3. FA2
                # --- Bucle de fusión, antes de FA2 ---
                centroA = (gridA[0] + gridA[-1]) / 2.0
                centroB = (gridB[0] + gridB[-1]) / 2.0

                fftA = aplicar_FA2(fftA, gridA - centroA, z_grid, Kc, dK_vec, z_ant, z_surf, eps_r)
                fftB = aplicar_FA2(fftB, gridB - centroB, z_grid, Kc, dK_vec, z_ant, z_surf, eps_r)

                # 4. FFT en azimuth (eje X)
                fftA_az = fft(fftA, axis=0)
                fftB_az = fft(fftB, axis=0)

                # --- Desplazar frecuencia cero al centro ---
                fftA_az_shift = np.fft.fftshift(fftA_az, axes=0)
                fftB_az_shift = np.fft.fftshift(fftB_az, axes=0)

                nxA, nz = fftA_az.shape
                nxB = fftB_az.shape[0]
                dxA = gridA[1] - gridA[0]
                dxB = gridB[1] - gridB[0]

                factor_A = compute_required_upsampling(fftA_az, dxA)
                factor_B = compute_required_upsampling(fftB_az, dxB)
                up_factor = max(factor_A, factor_B, up_factor_fallback)

                new_nx = int(max(nxA, nxB) * up_factor)
                new_nx = min(new_nx, 5000)

                # Zero-padding sobre espectros desplazados
                fftA_up = np.zeros((new_nx, nz), dtype=np.complex128)
                fftB_up = np.zeros((new_nx, nz), dtype=np.complex128)
                offA = (new_nx - nxA) // 2
                offB = (new_nx - nxB) // 2
                fftA_up[offA:offA+nxA, :] = fftA_az_shift
                fftB_up[offB:offB+nxB, :] = fftB_az_shift

                # Suma coherente
                fft_fused_shift = fftA_up + fftB_up

                #print(f"    Fusión {k//2}: max|fft_fused_shift| = {np.max(np.abs(fft_fused_shift)):.4e}")
                #print(f"    Fusión {k//2}: min|fft_fused_shift| = {np.min(np.abs(fft_fused_shift)):.4e}")

                # Deshacer desplazamiento para IFFT
                fft_fused = np.fft.ifftshift(fft_fused_shift, axes=0)

                # IFFT azimuth
                img_fused_az = ifft(fft_fused, axis=0)

                # IFFT rango
                img_fused = ifft(img_fused_az, axis=1)

                # Nueva grilla X
                x_min = min(gridA[0], gridB[0])
                x_max = max(gridA[-1], gridB[-1])
                new_grid = np.linspace(x_min, x_max, new_nx)

                new_images.append(img_fused)
                new_grids_x.append(new_grid)
            else:
                new_images.append(sub_images[k])
                new_grids_x.append(sub_grids_x[k])

        # Guardar frame (magnitud normalizada)
        if new_images:
            frame = np.abs(new_images[0])
            if np.max(frame) > 0:
                frame /= np.max(frame)
            frames.append(frame)

        sub_images = new_images
        sub_grids_x = new_grids_x
        nivel += 1

    final_img_complex = sub_images[0]
    final_x_grid = sub_grids_x[0]

    # Interpolación a malla objetivo si se solicita
    if target_x_grid is not None:
        interp_real = interp1d(final_x_grid, np.real(final_img_complex), axis=0,
                               kind='linear', bounds_error=False, fill_value=0)
        interp_imag = interp1d(final_x_grid, np.imag(final_img_complex), axis=0,
                               kind='linear', bounds_error=False, fill_value=0)
        img_interp_real = interp_real(target_x_grid)
        img_interp_imag = interp_imag(target_x_grid)
        final_img_complex = img_interp_real + 1j * img_interp_imag

    if return_complex:
        return final_img_complex, frames, final_x_grid
    else:
        final_img_mag = np.abs(final_img_complex)
        if np.max(final_img_mag) > 0:
            final_img_mag /= np.max(final_img_mag)
        return final_img_mag, frames, final_x_grid
    
def compute_K_from_ky(ky, eps_r):
    """
    Convierte frecuencia espacial ky (rad/m) a número de onda K (rad/m)
    usando la relación de fase estacionaria: K_y ≈ K * sqrt(eps_r).
    Artículo, Sec. II-B, discusión tras ec. (20).
    """
    # Para ky=0, K=0
    with np.errstate(divide='ignore', invalid='ignore'):
        K = np.where(ky == 0, 0, ky / np.sqrt(eps_r))
    return K

def aplicar_ventana_tukey(img, alpha=0.2):
    """Aplica ventana de Tukey 2D a la imagen compleja."""
    nx, nz = img.shape
    win_x = tukey(nx, alpha)
    win_z = tukey(nz, alpha)
    ventana = np.outer(win_x, win_z)
    return img * ventana

def compute_required_upsampling(fft_az, dx_current, energy_threshold=0.01, safe_margin=1.2):
    """
    Estima el factor de upsampling necesario para evitar aliasing en el dominio Kx.
    fft_az : espectro comprimido (nx, nz)
    dx_current : espaciado actual de la malla X (m)
    safe_margin : margen de seguridad (>1)
    Retorna factor entero recomendado.
    """
    nx, nz = fft_az.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx_current)
    energy_profile = np.sum(np.abs(fft_az)**2, axis=1)
    max_energy = np.max(energy_profile)
    if max_energy < 1e-15:
        return 2  # sin energía significativa, usar factor conservador
    mask = energy_profile > (energy_threshold * max_energy)
    indices = np.where(mask)[0]
    if len(indices) < 2:
        return 2
    kx_min = np.min(kx[indices])
    kx_max = np.max(kx[indices])
    bandwidth = kx_max - kx_min
    if bandwidth < 1e-6:  # evitar división por cero
        return 2
    dkx_current = 2 * np.pi / (nx * dx_current)
    dkx_required = 2 * np.pi / (bandwidth * safe_margin)
    factor = dkx_current / dkx_required
    # Limitar el factor a un máximo razonable (p.ej., 10)
    factor = max(1, min(int(np.ceil(factor)), 10))
    return factor

# =============================================================================
# 5. PROCESAMIENTO 3D: aplicar SCFBP a cada plano Y
# =============================================================================
def procesar_volumen_completo(data_3d_filt, x_ant_pos, y_ant_pos, tiempo,
                              X_IMG, Y_IMG, Z_IMG, z_ant, z_surf,
                              eps_r, suba_size, up_factor):
    """
    Procesa cada línea Y de antena con SCFBP‑2D y luego interpola el volumen
    a la malla Y_IMG deseada.

    Retorna volumen 3D con forma (len(X_IMG), len(Z_IMG), len(Y_IMG)).
    """
    from scipy.interpolate import interp1d

    ny_ant = len(y_ant_pos)                 # número de líneas Y escaneadas
    nx_img = len(X_IMG)
    nz_img = len(Z_IMG)

    # Array para almacenar los cortes X‑Z en las Y de antena
    slices_at_antenna = np.zeros((ny_ant, nx_img, nz_img), dtype=np.float64)

    for iy in range(ny_ant):
        y_actual = y_ant_pos[iy]
        print(f"Procesando línea Y = {y_actual:.3f} m ({iy+1}/{ny_ant})")

        # Extraer B‑scan para esta Y fija (todas las X)
        traces_2d = data_3d_filt[:, :, iy]          # shape (nt, nx_ant)

        # SCFBP 2D sobre esta línea
        img_2d, _, _ = scfbp_2d_completo(
            traces_2d,
            x_ant_pos,
            tiempo,
            X_IMG,
            Z_IMG,
            z_ant,
            z_surf,
            eps_r,
            suba_size=suba_size,
            up_factor_fallback=up_factor,          # <-- argumento nombrado
            return_complex=False,
            target_x_grid=X_IMG
        )

        slices_at_antenna[iy, :, :] = img_2d

    # Interpolación en Y desde las posiciones de antena a la malla Y_IMG
    volumen = np.zeros((nx_img, nz_img, len(Y_IMG)), dtype=np.float64)
    for ix in range(nx_img):
        for iz in range(nz_img):
            f_interp = interp1d(y_ant_pos, slices_at_antenna[:, ix, iz],
                                kind='linear', bounds_error=False,
                                fill_value='extrapolate')
            volumen[ix, iz, :] = f_interp(Y_IMG)

    return volumen

def visualizar_volumen_render(volumen, X_IMG, Y_IMG, Z_IMG, colormap='hot', 
                              umbral_min=0.1, umbral_max=0.9, opacity_scale=0.8):
    """
    Renderiza el volumen 3D con opacidad proporcional a la amplitud.
    
    Parámetros
    ----------
    volumen : ndarray (nx, nz, ny)
        Volumen 3D con valores de amplitud.
    X_IMG, Y_IMG, Z_IMG : ndarray
        Vectores de coordenadas en metros.
    colormap : str
        'gray', 'hot', 'viridis', 'plasma', 'inferno', etc.
    umbral_min : float
        Valor mínimo de amplitud (0-1) por debajo del cual los vóxeles son transparentes.
    umbral_max : float
        Valor máximo de amplitud (0-1) para saturación de color.
    opacity_scale : float
        Factor de escala para la opacidad (0-1). Valores mayores = más opaco.
    """
    import numpy as np
    
    # Normalizar el volumen entre 0 y 1
    v = np.abs(volumen.copy())
    if v.max() > 0:
        v = v / v.max()
    else:
        print("Advertencia: Volumen vacío (todos ceros)")
        return
    
    # Aplicar umbral mínimo para eliminar ruido de fondo
    v_display = np.where(v >= umbral_min, v, 0.0)
    
    # Mapear valores a opacidad de forma no lineal (más agresiva)
    # Fórmula: opacity = opacity_scale * ((v - umbral_min) / (1 - umbral_min))^2
    alpha = np.zeros_like(v_display)
    mask = v_display > umbral_min
    alpha[mask] = opacity_scale * ((v_display[mask] - umbral_min) / (1 - umbral_min)) ** 1.5
    
    # Transponer a (nx, ny, nz) para Plotly
    if volumen.shape == (len(X_IMG), len(Z_IMG), len(Y_IMG)):
        v_display = np.transpose(v_display, (0, 2, 1))
        alpha = np.transpose(alpha, (0, 2, 1))
    
    # Crear figura con renderizado volumétrico
    fig = go.Figure(data=go.Volume(
        x=X_IMG,
        y=Y_IMG,
        z=Z_IMG,
        value=v_display.flatten(),
        isomin=umbral_min,
        isomax=umbral_max,
        opacity=opacity_scale,              # opacidad base
        surface_count=25,                   # más superficies = mejor calidad
        colorscale=colormap,
        caps=dict(x_show=False, y_show=False, z_show=False),
        slices_z=dict(show=True, locations=[0.15]),  # corte en Z = 0.15m
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        title='Volumen 3D - Renderizado por transparencia',
        width=900,
        height=700
    )
    
    fig.show()
    
    # También mostrar cortes 2D para referencia
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Corte XY en Z central
    iz = len(Z_IMG) // 2
    axes[0].imshow(volumen[:, iz, :].T, extent=[X_IMG[0], X_IMG[-1], Y_IMG[0], Y_IMG[-1]],
                   origin='lower', cmap='hot', aspect='auto')
    axes[0].set_title(f'Corte XY en Z = {Z_IMG[iz]:.3f} m')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    
    # Corte XZ en Y central
    iy = len(Y_IMG) // 2
    axes[1].imshow(volumen[:, :, iy].T, extent=[X_IMG[0], X_IMG[-1], Z_IMG[0], Z_IMG[-1]],
                   origin='lower', cmap='hot', aspect='auto')
    axes[1].set_title(f'Corte XZ en Y = {Y_IMG[iy]:.3f} m')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    
    # Corte YZ en X central
    ix = len(X_IMG) // 2
    axes[2].imshow(volumen[ix, :, :].T, extent=[Z_IMG[0], Z_IMG[-1], Y_IMG[0], Y_IMG[-1]],
                   origin='lower', cmap='hot', aspect='auto')
    axes[2].set_title(f'Corte YZ en X = {X_IMG[ix]:.3f} m')
    axes[2].set_xlabel('Z (m)')
    axes[2].set_ylabel('Y (m)')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. VISUALIZACIÓN C‑SCAN (cortes horizontales y 3D)
# =============================================================================
def visualizar_cscan(volumen, X_IMG, Z_IMG, Y_IMG, z_slice=0.15, render_3d=True):
    """
    Muestra:
    - Un corte horizontal (C‑scan) a la profundidad z_slice.
    - Tres cortes ortogonales (opcional).
    """
    # Encontrar índice de profundidad más cercano
    iz = np.argmin(np.abs(Z_IMG - z_slice))
    slice_xz = volumen[:, :, Y_IMG.size//2]   # plano Y central
    slice_xy = volumen[:, iz, :]              # plano Z = z_slice

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

    np.save('volumen_3d.npy', volumen)

    if render_3d:
        visualizar_volumen_render(volumen, X_IMG, Y_IMG, Z_IMG, 
                                  colormap='hot', umbral_min=0.05, umbral_max=0.95, opacity_scale=0.9)

    '''
    # Visualización 3D básica (superficie de iso‑valor)
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        X, Y, Z = np.meshgrid(X_IMG, Y_IMG, Z_IMG, indexing='ij')
        # Extraer isosuperficie
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(volumen, level=0.3 * volumen.max())
        # Escalar vértices a coordenadas reales
        verts_scaled = np.zeros_like(verts)
        verts_scaled[:, 0] = np.interp(verts[:, 0], np.arange(len(X_IMG)), X_IMG)
        verts_scaled[:, 1] = np.interp(verts[:, 1], np.arange(len(Y_IMG)), Y_IMG)
        verts_scaled[:, 2] = np.interp(verts[:, 2], np.arange(len(Z_IMG)), Z_IMG)
        ax.plot_trisurf(verts_scaled[:, 0], verts_scaled[:, 1], faces, verts_scaled[:, 2],
                        cmap='viridis', alpha=0.8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Isosuperficie del bloque metálico (C‑scan 3D)')
        plt.show()
    except ImportError:
        print("Módulos para visualización 3D no disponibles (opcional).")
        '''
    

# ==========
# Fusion azimuth 4 ejes
# ======

def crear_interpolador_datos(data_3d, t_axis, x_axis, y_axis):
    """
    Crea un interpolador 3D para los datos (nt, nx, ny).
    Asume que data_3d es complejo; interpola parte real e imaginaria por separado.
    """
    nt, nx, ny = data_3d.shape
    # Ejes de la grilla original
    puntos = (t_axis, x_axis, y_axis)
    
    # Interpoladores para parte real e imaginaria
    interp_real = RegularGridInterpolator(puntos, data_3d.real, bounds_error=False, fill_value=0)
    interp_imag = RegularGridInterpolator(puntos, data_3d.imag, bounds_error=False, fill_value=0)
    
    def interpolador(t_query, x_query, y_query):
        pts = np.stack([t_query, x_query, y_query], axis=-1)
        return interp_real(pts) + 1j * interp_imag(pts)
    
    return interpolador

def generar_datos_diagonal(interpolador, t_axis, x_ant_orig, y_ant_orig,
                           direccion='45', n_trazas=30):
    """
    Genera un array de datos (nt, n_trazas) para una línea diagonal.
    direccion: '45' o '135'
    """
    # Definir línea de antenas en el plano XY
    if direccion == '45':
        # Línea a 45°: x = s, y = s  (desde (0.15,0.15) hasta (0.45,0.45))
        s_vals = np.linspace(X_START, X_END, n_trazas)
        x_line = s_vals
        y_line = s_vals
    elif direccion == '135':
        # Línea a 135°: x = s, y = (Y_END - (s - X_START))
        s_vals = np.linspace(X_START, X_END, n_trazas)
        x_line = s_vals
        y_line = Y_END - (s_vals - X_START)
    else:
        raise ValueError("Dirección no soportada")
    
    nt = len(t_axis)
    data_line = np.zeros((nt, n_trazas), dtype=np.complex128)
    
    # Para cada tiempo, interpolar en la línea espacial
    for i, (x, y) in enumerate(zip(x_line, y_line)):
        # Crear grid de consulta: todos los tiempos para una posición fija (x,y)
        t_grid, x_grid, y_grid = np.meshgrid(t_axis, [x], [y], indexing='ij')
        data_line[:, i] = interpolador(t_grid.ravel(), x_grid.ravel(), y_grid.ravel()).reshape(nt)
    
    return data_line, x_line, y_line

def procesar_plano_2d(data_2d, x_ant, tiempo, X_IMG, Z_IMG, z_ant, z_surf, eps_r,
                      suba_size, up_factor, target_x_grid=None):
    """
    Procesa un único B‑scan (adquisición 2D) con SCFBP‑2D.
    Retorna imagen 2D de magnitudes (nx, nz).
    """
    img_2d, _, _ = scfbp_2d_completo(
        data_2d, x_ant, tiempo,
        X_IMG, Z_IMG, z_ant, z_surf, eps_r,
        suba_size=suba_size,
        up_factor_fallback=up_factor,
        return_complex=False,
        target_x_grid=target_x_grid
    )
    return img_2d   # shape (len(X_IMG), len(Z_IMG))

# =========


def generate_synthetic_echo_complex(x_ant_pos, targets, z_ant, z_surf, eps_r,
                                    fc, bw, nt, dt, c=299792458.0):
    """
    Genera trazas complejas (señal analítica) mediante diseño en frecuencia.
    """
    n_ant = len(x_ant_pos)
    tiempo = np.arange(nt) * dt
    freq = np.fft.fftfreq(nt, d=dt)

    # Crear espectro de un pulso gaussiano paso bajo (banda base)
    sigma_f = bw / (2 * np.sqrt(2 * np.log(2)))  # ancho de banda a -3dB
    # Espectro gaussiano centrado en fc
    spectrum_pulse = np.exp(-0.5 * ((np.abs(freq) - fc) / sigma_f)**2)
    # Hacerlo cero en frecuencias negativas para obtener señal analítica
    spectrum_pulse[freq < 0] = 0

    traces = np.zeros((nt, n_ant), dtype=np.complex128)

    for i, xk in enumerate(x_ant_pos):
        spectrum = np.zeros(nt, dtype=np.complex128)
        for (xt, zt, amp) in targets:
            delay = tiempo_viaje(xk, xt, zt, z_ant, z_surf, eps_r)
            phase_shift = np.exp(-1j * 2 * np.pi * freq * delay)
            spectrum += amp * spectrum_pulse * phase_shift
        traces[:, i] = np.fft.ifft(spectrum)

    return traces, tiempo

def debug_scfbp_with_synthetic():
    print("\n=== DEBUG: Validación con ecos sintéticos (coherente) ===\n")

    c = 299792458.0
    z_ant = 0.35
    z_surf = 0.0
    eps_r_true = 6.0

    # Apertura más larga y densa
    x_ant = np.linspace(0.15, 0.45, 61)   # 61 posiciones

    targets = [(0.30, 0.15, 1.0)]

    fc = 1.5e9
    bw = 1.0e9
    nt = 4096
    dt = 2.5e-12

    print("Generando ecos sintéticos complejos...")
    traces_complex, tiempo = generate_synthetic_echo_complex(
        x_ant, targets, z_ant, z_surf, eps_r_true,
        fc, bw, nt, dt, c=c
    )

    print(f"  - Max |traces|: {np.max(np.abs(traces_complex)):.6f}")

    # Visualización B-scan (magnitud)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(tiempo * 1e9, np.abs(traces_complex[:, 30]))
    plt.xlabel('Tiempo (ns)')
    plt.ylabel('Amplitud')
    plt.title('Traza de ejemplo (magnitud)')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(traces_complex), aspect='auto',
               extent=[x_ant[0], x_ant[-1], tiempo[-1]*1e9, 0], cmap='gray')
    plt.xlabel('X antena (m)')
    plt.ylabel('Tiempo (ns)')
    plt.title('B‑scan sintético (magnitud)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Malla de imagen fina
    x_img = np.linspace(0.10, 0.50, 200)
    z_img = np.linspace(0.02, 0.30, 150)

    # BP directo coherente
    print("\nProbando BP directo coherente...")
    img_bp = bp_directo(traces_complex, x_ant, tiempo, x_img, z_img, z_ant, z_surf, eps_r_true)
    img_bp_mag = np.abs(img_bp)
    print(f"  - BP max magnitude: {np.max(img_bp_mag):.6f}")
    if np.max(img_bp_mag) < 1e-10:
        print("  ❌ BP directo vacío.")
        return

    img_bp_norm = img_bp_mag / np.max(img_bp_mag)

    plt.figure(figsize=(8, 6))
    plt.imshow(img_bp_norm.T, aspect='auto', extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]], cmap='gray')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.title('BP Directo (magnitud)')
    plt.plot(0.30, 0.15, 'ro', markersize=8, fillstyle='none')
    plt.colorbar()
    plt.show()

    # SCFBP coherente
    print("\nEjecutando SCFBP coherente...")
    try:
        img_scfbp, frames, x_final = scfbp_2d_completo(
            traces_complex, x_ant, tiempo,
            x_img, z_img, z_ant, z_surf, eps_r_true,
            suba_size=8,
            up_factor_fallback=2,
            return_complex=True,      # obtener imagen compleja
            target_x_grid=x_img
        )
        img_scfbp_mag = np.abs(img_scfbp)
        if np.max(img_scfbp_mag) > 0:
            img_scfbp_norm = img_scfbp_mag / np.max(img_scfbp_mag)
        else:
            print("  ⚠️ SCFBP produce imagen vacía")
            img_scfbp_norm = img_scfbp_mag
    except Exception as e:
        print(f"  ❌ Error en SCFBP: {e}")
        import traceback
        traceback.print_exc()
        return

    # Comparación
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_scfbp_norm.T, extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]], cmap='gray', aspect='auto')
    axes[0].set_title('SCFBP')
    axes[0].plot(0.30, 0.15, 'ro', markersize=8, fillstyle='none')
    axes[1].imshow(img_bp_norm.T, extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]], cmap='gray', aspect='auto')
    axes[1].set_title('BP Directo')
    axes[1].plot(0.30, 0.15, 'ro', markersize=8, fillstyle='none')
    diff = np.abs(img_scfbp_norm - img_bp_norm)
    im2 = axes[2].imshow(diff.T, extent=[x_img[0], x_img[-1], z_img[-1], z_img[0]], cmap='hot', aspect='auto')
    axes[2].set_title('Diferencia')
    plt.colorbar(im2, ax=axes[2])
    plt.tight_layout()
    plt.show()

    mse = np.mean((img_scfbp_norm - img_bp_norm)**2)
    print(f"\nMSE entre SCFBP y BP: {mse:.6f}")

    return img_scfbp_norm, img_bp_norm

# =============================================================================
# 7. SCRIPT PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Generar / cargar datos 3D
    # -------------------------------------------------------------------------
    print("=== GENERANDO C‑SCAN (simulaciones 3D) ===")
    x_ant_pos, y_ant_pos, archivos_out = generar_cscan()

    print("=== CARGANDO DATOS 3D ===")
    tiempo, dt, nt, data_3d = cargar_datos_3d(archivos_out)
  
    '''
    # --- DIAGNÓSTICO DE TIEMPO ---
    print(f"  dt = {dt:.3e} s")
    print(f"  nt = {nt}")
    print(f"  tiempo[0] = {tiempo[0]:.3e} s")
    print(f"  tiempo[-1] = {tiempo[-1]:.3e} s ({tiempo[-1]*1e9:.2f} ns)")
    print(f"  Frecuencia de muestreo = {1/dt:.3e} Hz")
    # -----------------------------

    # Después de cargar
    plt.figure(figsize=(10, 4))
    plt.plot(tiempo * 1e9, data_3d[:, N_TRAZAS_X//2, N_TRAZAS_Y//2].real)
    plt.xlabel('Tiempo (ns)')
    plt.ylabel('Amplitud')
    plt.title('Traza central (parte real)')
    plt.grid(True)
    plt.show()
    #------------------------------
    x_centro = X_IMG[len(X_IMG)//2]
    z_centro = 0.15   # profundidad del bloque
    t_teorico = tiempo_viaje(x_ant_pos[N_TRAZAS_X//2], x_centro, z_centro, Z_ANT, Z_SURFACE, EPS_R)
    print(f"Tiempo de viaje teórico al centro del bloque: {t_teorico*1e9:.3f} ns")
    '''
    print("=== APLICANDO FILTROS Y ENVOLVENTE ===")
    data_filtrada_3d = aplicar_filtros_3d(data_3d, dt)

    # -------------------------------------------------------------------------
    # 2. Inversión de permitividad (sobre plano Y central)
    # -------------------------------------------------------------------------
    iy_central = len(y_ant_pos) // 2
    traces_central = data_filtrada_3d[:, :, iy_central]
    print(f"\n=== ESTIMACIÓN DE PERMITIVIDAD (plano Y={y_ant_pos[iy_central]:.3f}) ===")

    # Permitividad inicial supuesta (errónea)
    eps_guess = EPS_R

    # Ejecutar SCFBP con la permitividad errónea (devuelve imagen compleja)
    img_complex, _, x_final = scfbp_2d_completo(
        traces_central, x_ant_pos, tiempo,
        X_IMG, Z_IMG, Z_ANT, Z_SURFACE, eps_guess,
        suba_size=SUBA_PER_SIZE,
        up_factor_fallback=UPSAMPLE_FACTOR_FALLBACK,
        return_complex=True,
        target_x_grid=None
    )

    # Cálculos para la inversión (usando la geometría corregida)
    fc = FRECUENCIA_CENTRAL_PULSO
    # fc = (BANDA_PASA[0] + BANDA_PASA[1]) / 2
    Kc = 4.0 * np.pi * fc / c
    ky = 2.0 * np.pi * fftfreq(len(Z_IMG), d=dz)          # factor 2π añadido
    K = compute_K_from_ky(ky, eps_guess)                  # K = ky / sqrt(eps_r)
    dK_vec = K - Kc

    z_ref = 0.15   # profundidad del blanco esperado
    p, _, _ = estimate_phase_error_from_image(
        img_complex, x_final, Z_IMG, z_ref, Kc, dK_vec, Z_ANT, Z_SURFACE, eps_guess
    )

    # Invertir permitividad (puede descomentar cuando desee usarla)
    # eps_inverted = invert_permittivity_from_p(
    #     p, Kc, x_ref=0.30, z_ref=z_ref, z_ant=Z_ANT, z_surf=Z_SURFACE, eps_guess=eps_guess
    # )
    eps_inverted = 1.0   # temporalmente fijo (cámbielo por el valor real de su modelo)
    print(f"Permitividad invertida: ε_r = {eps_inverted:.3f} (real del modelo: {EPS_R})")

    '''

    # -------------------------------------------------------------------------
    # 3. Procesar todo el volumen con la permitividad corregida
    # -------------------------------------------------------------------------
    print("\n=== PROCESANDO VOLUMEN 3D COMPLETO ===")
    #t_start = time.perf_counter()
    volumen_3d = procesar_volumen_completo(
        data_filtrada_3d, x_ant_pos, y_ant_pos, tiempo,
        X_IMG, Y_IMG, Z_IMG, Z_ANT, Z_SURFACE,
        eps_inverted, SUBA_PER_SIZE, UPSAMPLE_FACTOR_FALLBACK
    )
    #t_total = time.perf_counter() - t_start
    #print(f"Tiempo total de procesamiento 3D: {t_total:.2f} s")

    # -------------------------------------------------------------------------
    # 4. Visualizar resultados
    # -------------------------------------------------------------------------
    visualizar_cscan(volumen_3d, X_IMG, Z_IMG, Y_IMG, z_slice=0.15)

    print("¡Procesamiento C‑scan completado!")
    '''

    # -------------------------------------------------------------------------
    # 3a. Procesar volumen con acimut en X (planos X‑Z para cada Y)
    # -------------------------------------------------------------------------
    print("\n=== PROCESANDO VOLUMEN X‑Z (acimut en X) ===")
    volumen_XZ = procesar_volumen_completo(
        data_filtrada_3d, x_ant_pos, y_ant_pos, tiempo,
        X_IMG, Y_IMG, Z_IMG, Z_ANT, Z_SURFACE,
        eps_inverted, SUBA_PER_SIZE, UPSAMPLE_FACTOR_FALLBACK
    )   # forma: (len(X_IMG), len(Z_IMG), len(Y_IMG))

    # -------------------------------------------------------------------------
    # 3b. Procesar volumen con acimut en Y (planos Y‑Z para cada X)
    # -------------------------------------------------------------------------
    print("\n=== PROCESANDO VOLUMEN Y‑Z (acimut en Y) ===")
    # Transponer los datos: (nt, nx, ny) -> (nt, ny, nx)
    data_filtrada_3d_T = np.transpose(data_filtrada_3d, (0, 2, 1))

    # Ahora 'x_ant_pos' e 'y_ant_pos' intercambian roles
    volumen_YZ_raw = procesar_volumen_completo(
        data_filtrada_3d_T, y_ant_pos, x_ant_pos, tiempo,
        Y_IMG, X_IMG, Z_IMG, Z_ANT, Z_SURFACE,
        eps_inverted, SUBA_PER_SIZE, UPSAMPLE_FACTOR_FALLBACK
    )   # forma: (len(Y_IMG), len(Z_IMG), len(X_IMG))

    # Transponer para que tenga la misma orientación que volumen_XZ: (nx, nz, ny)
    volumen_YZ = np.transpose(volumen_YZ_raw, (2, 1, 0))

    # -------------------------------------------------------------------------
    # 3c. Procesar plano diagonal 45°
    # -------------------------------------------------------------------------
    print("\n=== PROCESANDO PLANO DIAGONAL 45° ===")

    # Crear interpolador de los datos filtrados
    interp = crear_interpolador_datos(data_filtrada_3d, tiempo, x_ant_pos, y_ant_pos)

    # Generar datos a lo largo de la línea a 45°
    data_45, x_line_45, y_line_45 = generar_datos_diagonal(
        interp, tiempo, x_ant_pos, y_ant_pos, direccion='45', n_trazas=N_TRAZAS_X
    )

    # Coordenada de acimut: distancia a lo largo de la línea diagonal
    s_acimut = np.sqrt((x_line_45 - X_START)**2 + (y_line_45 - Y_START)**2)

    # Procesar el plano 2D con SCFBP
    img_D1 = procesar_plano_2d(
        data_45, s_acimut, tiempo,
        X_IMG, Z_IMG, Z_ANT, Z_SURFACE, eps_inverted,
        SUBA_PER_SIZE, UPSAMPLE_FACTOR_FALLBACK, target_x_grid=X_IMG
    )   # shape: (nx, nz)

    # Expandir a volumen 3D replicando a lo largo de Y
    volumen_D1 = np.repeat(img_D1[:, :, np.newaxis], len(Y_IMG), axis=2)  # (nx, nz, ny)

    # -------------------------------------------------------------------------
    # 3d. Procesar plano diagonal 135°
    # -------------------------------------------------------------------------
    print("\n=== PROCESANDO PLANO DIAGONAL 135° ===")
    data_135, x_line_135, y_line_135 = generar_datos_diagonal(
        interp, tiempo, x_ant_pos, y_ant_pos, direccion='135', n_trazas=N_TRAZAS_X
    )
    s_acimut_135 = np.sqrt((x_line_135 - X_START)**2 + (y_line_135 - Y_START)**2)

    img_D2 = procesar_plano_2d(
        data_135, s_acimut_135, tiempo,
        X_IMG, Z_IMG, Z_ANT, Z_SURFACE, eps_inverted,
        SUBA_PER_SIZE, UPSAMPLE_FACTOR_FALLBACK, target_x_grid=X_IMG
    )
    volumen_D2 = np.repeat(img_D2[:, :, np.newaxis], len(Y_IMG), axis=2)

    # -------------------------------------------------------------------------
    # 4. Fusión de los cuatro volúmenes
    # -------------------------------------------------------------------------
    print("\n=== FUSIONANDO CUATRO VOLÚMENES ===")

    # Normalizar cada volumen
    vol_XZ_norm = volumen_XZ / np.max(volumen_XZ)
    vol_YZ_norm = volumen_YZ / np.max(volumen_YZ)
    vol_D1_norm = volumen_D1 / np.max(volumen_D1)
    vol_D2_norm = volumen_D2 / np.max(volumen_D2)

    # Multiplicación punto a punto
    volumen_fused = vol_XZ_norm * vol_YZ_norm * vol_D1_norm * vol_D2_norm

    # Visualizar resultado
    visualizar_cscan(volumen_fused, X_IMG, Z_IMG, Y_IMG, z_slice=0.15)

    print("¡Procesamiento C‑scan con fusión ortogonal completado!")


    #debug_scfbp_with_synthetic()