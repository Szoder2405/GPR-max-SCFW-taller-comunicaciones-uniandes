#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def visualizar_volumen_isosurface(volumen, X_IMG, Y_IMG, Z_IMG, 
                                  nivel=None, colormap='hot', opacity=0.8):
    """
    Muestra una isosuperficie del volumen (superficie de nivel constante).
    """
    v = np.abs(volumen).copy()
    v_max = v.max()
    if v_max == 0:
        print("Volumen vacío.")
        return
    v_norm = v / v_max

    if nivel is None:
        nivel = 0.5

    # Crear grid 3D con indexing='ij' para que la forma sea (nx, ny, nz)
    X, Y, Z = np.meshgrid(X_IMG, Y_IMG, Z_IMG, indexing='ij')
    # Transponer volumen a (nx, ny, nz) si es necesario
    if v_norm.shape == (len(X_IMG), len(Z_IMG), len(Y_IMG)):
        v_norm = np.transpose(v_norm, (0, 2, 1))

    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=v_norm.flatten(),
        isomin=nivel,
        isomax=nivel,
        opacity=opacity,
        surface_count=1,
        colorscale=colormap,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        title=f'Isosuperficie (nivel = {nivel:.2f} del máximo)',
        width=900,
        height=700
    )
    fig.show()


def visualizar_volumen_scatter_fixed(volumen, X_IMG, Y_IMG, Z_IMG, 
                                     umbral_percentil=70, colormap='gray', 
                                     punto_size=15, opacity=0.001):
    """
    Nube de puntos 3D con opacidad FIJA. Los colores se mantienen fieles
    al valor de amplitud (sin oscurecerse por el umbral).
    """
    v = np.abs(volumen).copy()
    v_max = v.max()
    if v_max == 0:
        print("Volumen vacío.")
        return
    v_norm = v / v_max

    umbral = np.percentile(v_norm, umbral_percentil)
    print(f"Umbral calculado (percentil {umbral_percentil}): {umbral:.3f}")

    # Transponer volumen a (nx, ny, nz) si es necesario
    if v_norm.shape == (len(X_IMG), len(Z_IMG), len(Y_IMG)):
        v_norm = np.transpose(v_norm, (0, 2, 1))
        print("Volumen transpuesto a (nx, ny, nz)")

    # Crear grid 3D
    X, Y, Z = np.meshgrid(X_IMG, Y_IMG, Z_IMG, indexing='ij')

    mask = v_norm >= umbral
    x_plot = X[mask]
    y_plot = Y[mask]
    z_plot = Z[mask]
    v_plot = v_norm[mask]

    print(f"Puntos a mostrar: {len(x_plot)}")

    if len(x_plot) == 0:
        print("Ningún vóxel supera el umbral. Reduzca el percentil.")
        return

    # =========================================================================
    # SOLUCIÓN: Forzar el rango de color al mínimo y máximo de los datos visibles
    # =========================================================================
    cmin = v_plot.min()  # Mínimo valor visible (será >= umbral)
    cmax = v_plot.max()  # Máximo valor visible (normalmente 1.0)

    fig = go.Figure(data=go.Scatter3d(
        x=x_plot,
        y=y_plot,
        z=z_plot,
        mode='markers',
        marker=dict(
            size=punto_size,
            color=v_plot,
            colorscale=colormap + '_r', # <-- AÑADIR '_r' PARA INVERTIR
            cmin=cmin,           # <-- El color más bajo de la escala se asigna a cmin
            cmax=cmax,           # <-- El color más alto se asigna a cmax
            opacity=opacity,
            colorbar=dict(title='Amplitud norm.')
        ),
        text=[f'Amp: {val:.3f}' for val in v_plot],
        hoverinfo='text'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        title=f'Nube de puntos (amplitud > percentil {umbral_percentil})',
        width=900,
        height=700
    )
    fig.show()


def visualizar_volumen_render_auto(volumen, X_IMG, Y_IMG, Z_IMG, 
                                   colormap='hot', umbral_percentil=50):
    """
    Render volumétrico con umbral automático basado en percentil.
    Versión mejorada para garantizar visibilidad.
    """
    v = np.abs(volumen).copy()
    v_max = v.max()
    if v_max == 0:
        print("Volumen vacío.")
        return
    v_norm = v / v_max

    umbral_min = np.percentile(v_norm[v_norm > 0], umbral_percentil) if np.any(v_norm > 0) else 0.1
    print(f"Umbral mínimo (percentil {umbral_percentil}): {umbral_min:.4f}")

    # Transponer a (nx, ny, nz) si es necesario
    if v_norm.shape == (len(X_IMG), len(Z_IMG), len(Y_IMG)):
        v_norm = np.transpose(v_norm, (0, 2, 1))
        print(f"Volumen transpuesto a: {v_norm.shape} (nx, ny, nz)")
    
    # Crear figura con opacidad baja y muchos detalles
    fig = go.Figure(data=go.Volume(
        x=X_IMG,
        y=Y_IMG,
        z=Z_IMG,
        value=v_norm.flatten(),
        isomin=umbral_min,
        isomax=1.0,
        opacity=0.08,               # <-- Muy bajo para ver a través
        surface_count=30,           # <-- Más superficies = mejor calidad
        colorscale=colormap,
        caps=dict(x_show=False, y_show=False, z_show=False),
        lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),  # Mejor iluminación
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        title=f'Render volumétrico (umbral percentil {umbral_percentil})',
        width=900,
        height=700
    )
    fig.show()


if __name__ == "__main__":
    #archivo = 'volumen_scfbp_3d_FA1.npy'
    archivo = 'volumen_3d.npy'
    if len(sys.argv) > 1:
        archivo = sys.argv[1]

    try:
        vol = np.load(archivo)
        print(f"Volumen cargado: {vol.shape} (nx, nz, ny)")
    except FileNotFoundError:
        print(f"Archivo '{archivo}' no encontrado.")
        sys.exit(1)

    nx, nz, ny = vol.shape
    X_IMG = np.linspace(0.15, 0.45, nx)
    Y_IMG = np.linspace(0.15, 0.45, ny)
    Z_IMG = np.linspace(0.05, 0.35, nz)

    print(f"Rango de amplitudes: [{vol.min():.3e}, {vol.max():.3e}]")

    print("\nSeleccione tipo de visualización:")
    print("1. Isosuperficie (recomendado)")
    print("2. Nube de puntos (scatter)")
    print("3. Render volumétrico")
    opcion = input("Opción (1/2/3): ").strip()

    if opcion == '1':
        nivel = input("Nivel de isosuperficie (0-1, Enter = 0.5): ").strip()
        nivel = float(nivel) if nivel else 0.5
        visualizar_volumen_isosurface(vol, X_IMG, Y_IMG, Z_IMG, nivel=nivel)
    elif opcion == '2':
        perc = input("Percentil de umbral (Enter = 70): ").strip()
        perc = float(perc) if perc else 70.0
        visualizar_volumen_scatter_fixed(vol, X_IMG, Y_IMG, Z_IMG, umbral_percentil=perc)
    elif opcion == '3':
        perc = input("Percentil para umbral mínimo (Enter = 50): ").strip()
        perc = float(perc) if perc else 50.0
        visualizar_volumen_render_auto(vol, X_IMG, Y_IMG, Z_IMG, umbral_percentil=perc)
    else:
        print("Opción no válida.")