#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procesa en lote todos los A-scans desde una carpeta de salida de gprMax.
Genera una figura completa (voltajes, corrientes, S11, potencia, espectro)
y guarda datos crudos en .npz.

Uso:
    python batch_ascans.py [carpeta_entrada] [carpeta_salida]
Argumentos:
    carpeta_entrada : Carpeta que contiene los archivos .out (por defecto: temp_cscan_output)
    carpeta_salida  : Carpeta donde se guardarán los resultados (por defecto: Ascan_analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
from scipy.fft import fft, fftfreq

# =============================================================================
# Parámetros por defecto
# =============================================================================
DEFAULT_INPUT_DIR = "temp_cscan_output"
DEFAULT_OUTPUT_DIR = "Ascan_analysis"
N_X = 30                 # número de posiciones en X (0..30)
N_Y = 30                 # número de posiciones en Y (0..30)
Z0 = 50.0                # impedancia de la línea

# =============================================================================
# Funciones de carga y cálculo
# =============================================================================
def cargar_ascan(archivo):
    with h5py.File(archivo, 'r') as f:
        dt = f.attrs['dt']
        nt = f.attrs['Iterations']
        tiempo = np.arange(nt) * dt * 1e9          # ns
        Vinc = f['tls']['tl1']['Vinc'][:]
        Vtotal = f['tls']['tl1']['Vtotal'][:]
        Vref = Vtotal - Vinc
        Iinc = Vinc / Z0
        Iref = Vref / Z0
        Itotal = Vtotal / Z0
    return tiempo, dt, nt, Vinc, Vtotal, Vref, Iinc, Iref, Itotal

def calcular_s11(Vinc, Vref, dt):
    nt = len(Vinc)
    freq = fftfreq(nt, d=dt) / 1e9
    Vinc_f = fft(Vinc)
    Vref_f = fft(Vref)
    eps = 1e-10
    S11 = np.where(np.abs(Vinc_f) > eps, Vref_f / Vinc_f, 0)
    S11_db = 20 * np.log10(np.abs(S11) + 1e-15)
    mask_pos = freq >= 0
    return freq[mask_pos], S11_db[mask_pos], S11[mask_pos]

def crear_figura_resumen(tiempo, Vinc, Vref, Vtotal,
                         Iinc, Iref, Itotal,
                         Pinc, Pref,
                         freq_pos, Vinc_f_pos, Vref_f_pos,
                         S11_db, fase_s11):
    """Crea una figura 5x3 con todos los análisis."""
    fig, axes = plt.subplots(5, 3, figsize=(22, 22))
    fig.suptitle('Análisis completo A‑scan', fontsize=18, fontweight='bold')

    # --- Fila 1: Voltajes incidente, reflejado, total ---
    ax = axes[0, 0]
    ax.plot(tiempo, Vinc, 'b', lw=1)
    ax.set_ylabel('V')
    ax.set_title('$V_{inc}$')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(15, tiempo[-1]))

    ax = axes[0, 1]
    ax.plot(tiempo, Vref, 'r', lw=1)
    ax.set_ylabel('V')
    ax.set_title('$V_{ref}$')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(15, tiempo[-1]))

    ax = axes[0, 2]
    ax.plot(tiempo, Vtotal, 'g', lw=1)
    ax.set_ylabel('V')
    ax.set_title('$V_{total}$')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(15, tiempo[-1]))

    # --- Fila 2: Superposición voltajes, Corrientes incidente, reflejada ---
    ax = axes[1, 0]
    ax.plot(tiempo, Vinc, 'b', lw=0.9, alpha=0.7, label='$V_{inc}$')
    ax.plot(tiempo, Vref, 'r', lw=0.9, alpha=0.7, label='$V_{ref}$')
    ax.plot(tiempo, Vtotal, 'g', lw=0.9, alpha=0.7, label='$V_{total}$')
    ax.set_ylabel('V')
    ax.set_title('Superposición de voltajes')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(15, tiempo[-1]))

    ax = axes[1, 1]
    ax.plot(tiempo, Iinc*1000, 'b', lw=1)
    ax.set_ylabel('mA')
    ax.set_title('$I_{inc}$')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(15, tiempo[-1]))

    ax = axes[1, 2]
    ax.plot(tiempo, Iref*1000, 'r', lw=1)
    ax.set_ylabel('mA')
    ax.set_title('$I_{ref}$')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(15, tiempo[-1]))

    # --- Fila 3: Superposición corrientes, Potencia incidente, reflejada ---
    ax = axes[2, 0]
    ax.plot(tiempo, Iinc*1000, 'b', lw=0.9, alpha=0.7, label='$I_{inc}$')
    ax.plot(tiempo, Iref*1000, 'r', lw=0.9, alpha=0.7, label='$I_{ref}$')
    ax.plot(tiempo, Itotal*1000, 'g', lw=0.9, alpha=0.7, label='$I_{total}$')
    ax.set_ylabel('mA')
    ax.set_title('Superposición de corrientes')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(15, tiempo[-1]))

    ax = axes[2, 1]
    ax.plot(tiempo, Pinc*1000, 'b', lw=1, label='$P_{inc}$')
    ax.set_ylabel('mW')
    ax.set_title('Potencia incidente')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(15, tiempo[-1]))

    ax = axes[2, 2]
    ax.plot(tiempo, -Pref*1000, 'r', lw=1, label='$-P_{ref}$')
    ax.set_ylabel('mW')
    ax.set_title('Potencia reflejada (invertida)')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, min(15, tiempo[-1]))

    # --- Fila 4: Espectros normalizados, S11 magnitud, S11 fase ---
    ax = axes[3, 0]
    ax.plot(freq_pos, Vinc_f_pos/np.max(Vinc_f_pos+1e-12), 'b', lw=1.2, label='$V_{inc}$')
    ax.plot(freq_pos, Vref_f_pos/np.max(Vref_f_pos+1e-12), 'r', lw=1.2, label='$V_{ref}$')
    ax.set_xlabel('Frecuencia (GHz)')
    ax.set_ylabel('Amplitud norm.')
    ax.set_title('Espectro normalizado')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 5)

    ax = axes[3, 1]
    ax.plot(freq_pos, S11_db, 'm', lw=1.2)
    ax.set_xlabel('Frecuencia (GHz)')
    ax.set_ylabel('|S11| (dB)')
    ax.set_title('$S_{11}$')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(-60, 10)
    ax.axhline(y=0, color='k', ls='--', alpha=0.3)

    ax = axes[3, 2]
    ax.plot(freq_pos, fase_s11, 'c', lw=1.2)
    ax.set_xlabel('Frecuencia (GHz)')
    ax.set_ylabel('Fase (°)')
    ax.set_title('Fase $S_{11}$')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 5)

    # --- Fila 5: Vacía o con resumen numérico (aquí dejamos solo un texto) ---
    for j in range(3):
        axes[4, j].axis('off')
    # Se puede añadir un texto con parámetros clave
    texto_resumen = (
        f"dt = {tiempo[1]-tiempo[0]:.3e} ns  |  "
        f"Máx |Vinc| = {np.max(np.abs(Vinc)):.3f} V  |  "
        f"Máx |Vref| = {np.max(np.abs(Vref)):.3f} V\n"
        f"Pico Vref en t = {tiempo[np.argmax(np.abs(Vref))]:.2f} ns"
    )
    fig.text(0.5, 0.02, texto_resumen, ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig

# =============================================================================
# Procesamiento en lote
# =============================================================================
def main():
    if len(sys.argv) >= 2:
        input_dir = sys.argv[1]
    else:
        input_dir = DEFAULT_INPUT_DIR

    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    print(f"Directorio de entrada : {input_dir}")
    print(f"Directorio de salida  : {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    total = N_X * N_Y
    contador = 0
    for iy in range(N_Y):
        for ix in range(N_X):
            filename = f"temp_{ix:03d}_{iy:03d}.out"
            filepath = os.path.join(input_dir, filename)
            if not os.path.exists(filepath):
                continue

            contador += 1
            print(f"Procesando {contador}/{total}: {filename}")

            # Cargar
            tiempo, dt, nt, Vinc, Vtotal, Vref, Iinc, Iref, Itotal = cargar_ascan(filepath)

            # Calcular S11 y espectros
            freq_pos, S11_db, S11_complex = calcular_s11(Vinc, Vref, dt)
            Vinc_f = np.abs(fft(Vinc))
            Vref_f = np.abs(fft(Vref))
            mask_pos = fftfreq(nt, d=dt) >= 0
            Vinc_f_pos = Vinc_f[mask_pos]
            Vref_f_pos = Vref_f[mask_pos]
            fase_s11 = np.angle(S11_complex, deg=True)

            # Potencias
            Pinc = Vinc * Iinc
            Pref = Vref * Iref

            # Crear figura
            fig = crear_figura_resumen(tiempo, Vinc, Vref, Vtotal,
                                       Iinc, Iref, Itotal,
                                       Pinc, Pref,
                                       freq_pos, Vinc_f_pos, Vref_f_pos,
                                       S11_db, fase_s11)

            # Guardar imagen
            png_name = f"ascan_{ix:03d}_{iy:03d}.png"
            png_path = os.path.join(output_dir, png_name)
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Guardar datos crudos
            npz_name = f"ascan_{ix:03d}_{iy:03d}.npz"
            npz_path = os.path.join(output_dir, npz_name)
            np.savez(npz_path,
                     tiempo=tiempo, dt=dt, nt=nt,
                     Vinc=Vinc, Vtotal=Vtotal, Vref=Vref,
                     Iinc=Iinc, Iref=Iref, Itotal=Itotal,
                     Pinc=Pinc, Pref=Pref,
                     freq=freq_pos, S11_db=S11_db, fase_s11=fase_s11,
                     Vinc_f_pos=Vinc_f_pos, Vref_f_pos=Vref_f_pos)

    print(f"\nProcesamiento completado. {contador} archivos guardados en '{output_dir}'.")

if __name__ == "__main__":
    main()