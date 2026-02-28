import h5py
import numpy as np
import vtk
from vtk.util import numpy_support

# Parámetros (deben coincidir con tu simulación)
x_inicio = 0.1
y_inicio = 0.1
paso = 0.005
nx = 61   # (0.4-0.1)/0.005 + 1
ny = 61

# Cargar datos del archivo HDF5
with h5py.File('C:/Users/santi/gprMax/resultados_cscan/scan_merged.out', 'r') as f:
    # Normalmente los datos están en /rxs/rx1/Ex (o Ez). Ajusta según tu receptor.
    # Si usaste #rx sin especificar, puede tener múltiples componentes. Elige la que quieras.
    data = f['rxs']['rx1']['Ez'][:]   # forma: (tiempo, ny, nx)

# Dimensiones
nt, ny, nx = data.shape

# Crear imagen VTK
imageData = vtk.vtkImageData()
imageData.SetDimensions(nx, ny, nt)
imageData.SetSpacing(paso, paso, 5e-9 / (nt-1))   # espaciado en X, Y y tiempo (en segundos)
imageData.SetOrigin(x_inicio, y_inicio, 0)

# Convertir datos a formato VTK (orden Fortran para que coincida)
flat_data = data.flatten(order='F')
vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)
vtk_array.SetName('Ez')
imageData.GetPointData().SetScalars(vtk_array)

# Escribir archivo .vti
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName('C:/Users/santi/gprMax/resultados_cscan/cscan.vti')
writer.SetInputData(imageData)
writer.Write()

print("Archivo cscan.vti generado. Ábrelo con ParaView.")