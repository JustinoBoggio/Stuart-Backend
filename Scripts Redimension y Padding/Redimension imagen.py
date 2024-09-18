import cv2
import os

# Directorio de imágenes
# input_directory = './DeteccionFondoCaja/test_padding'
# output_directory = './DeteccionFondoCaja/test_padding_resized'

input_directory = './DeteccionObjetosInteres/train_padding'
output_directory = './DeteccionObjetosInteres/train_padding_resized'

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
    
    resized_frame = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)

    # Crear el subdirectorio de salida si no existe
    output_path = os.path.join(output_directory, image_file)
    output_subdir = os.path.dirname(output_path)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    # Guardar la imagen procesada
    if cv2.imwrite(output_path, resized_frame):
        print(f"Imagen guardada: {output_path}")
    else:
        print(f"No se pudo guardar la imagen: {output_path}")