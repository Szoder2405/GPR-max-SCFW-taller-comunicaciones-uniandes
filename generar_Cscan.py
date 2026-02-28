import os
import subprocess
import shutil

# Parámetros
x_inicio = 0.1
x_fin = 0.4
y_inicio = 0.1
y_fin = 0.4
paso = 0.005
nx = int((x_fin - x_inicio)/paso) + 1
ny = int((y_fin - y_inicio)/paso) + 1

# Crear carpeta para los archivos temporales y resultados
os.makedirs("temp_cscan", exist_ok=True)
os.makedirs("resultados_cscan", exist_ok=True)

# Plantilla del archivo .in
plantilla = """#domain: 0.5 0.5 0.5
#dx_dy_dz: 0.005 0.005 0.005
#time_window: 5e-9

#material: 6 0 1 0 material_A
#material: 2 0 1 0 material_B

#box: 0.10 0.10 0.10 0.20 0.20 0.20 material_A
#box: 0.30 0.10 0.10 0.40 0.20 0.20 material_B
#box: 0.10 0.30 0.10 0.20 0.40 0.20 material_A
#box: 0.30 0.30 0.10 0.40 0.40 0.20 material_B

#waveform: ricker 1 1.5e9 mi_pulso
#hertzian_dipole: z {x_pos} {y_pos} 0.45 mi_pulso
#rx: {x_pos} {y_pos} 0.05
"""

# Directorio de trabajo de gprMax (ajusta si es necesario)
os.chdir(r"C:\Users\santi\gprMax")

# Ejecutar para cada posición
for i, x in enumerate([x_inicio + i*paso for i in range(nx)]):
    for j, y in enumerate([y_inicio + j*paso for j in range(ny)]):
        nombre_in = f"temp_cscan/scan_{i:03d}_{j:03d}.in"
        with open(nombre_in, "w") as f:
            f.write(plantilla.format(x_pos=x, y_pos=y))
        
        # Ejecutar gprMax
        resultado = subprocess.run(
            f"python -m gprMax {nombre_in}",
            shell=True,
            capture_output=True,
            text=True
        )
        if resultado.returncode != 0:
            print(f"Error en posición ({x:.3f},{y:.3f}):")
            print(resultado.stderr)
        else:
            # Mover el archivo de salida a la carpeta de resultados
            archivo_out = nombre_in.replace('.in', '.out')
            if os.path.exists(archivo_out):
                shutil.move(archivo_out, f"resultados_cscan/scan_{i:03d}_{j:03d}.out")
        
        # Opcional: borrar el archivo .in
        os.remove(nombre_in)

print("Simulaciones completadas.")
print(f"Archivos de salida en resultados_cscan/")