import cv2
import os
import numpy as np

# Definir el tamaño deseado (por ejemplo, cuadrado 1920x1920)
desired_size = 1920

# Directorio de imágenes
# input_directory = './test'
# output_directory = './DeteccionFondoCaja/test_padding'

input_directory = './DeteccionObjetosInteres/test'
output_directory = './DeteccionObjetosInteres/test_padding'

# Crear el directorio de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Obtener la lista de archivos en el directorio
image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Procesar cada imagen en el directorio
for image_file in image_files:

    # Cargar la imagen
    image_path = os.path.join(input_directory, image_file)
    image = cv2.imread(image_path)

    # Verificar si la imagen se cargó correctamente
    if image is None:
        print(f"No se pudo cargar la imagen: {image_file}")
        continue

    # Obtener el tamaño actual de la imagen
    old_size = image.shape[:2]  # (height, width)

    # Calcular la cantidad de padding necesario
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Aplicar padding a la imagen
    color = [0, 0, 0]  # Color del padding (negro en este caso)
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Crear el subdirectorio de salida si no existe
    output_path = os.path.join(output_directory, image_file)
    output_subdir = os.path.dirname(output_path)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    # Guardar la imagen procesada
    if cv2.imwrite(output_path, new_image):
        print(f"Imagen guardada: {output_path}")
    else:
        print(f"No se pudo guardar la imagen: {output_path}")