# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Cargar el video
# video_path = 'D:/Leonel/ISI/Tesis/OneDrive_2024-05-23/NOR hembras escopolamina completo/G2/Reconocimiento_2_G.mp4'
# cap = cv2.VideoCapture(video_path)

# # Verificar si el video se abrió correctamente
# if not cap.isOpened():
#     print("Error al abrir el video")
# else:
#     print("Video abierto exitosamente")

# positions = []

# # Obtener las dimensiones de la pantalla
# screen_width = 800
# screen_height = 600

# # Función para redimensionar el frame
# def resize_frame(frame, max_width, max_height):
#     (h, w) = frame.shape[:2]
#     if w > max_width or h > max_height:
#         # Calcular la proporción de redimensionamiento
#         ratio = min(max_width / float(w), max_height / float(h))
#         dim = (int(w * ratio), int(h * ratio))
#         resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#     else:
#         resized = frame
#     return resized

# # Función para detectar la caja en el primer frame
# def detectar_caja(frame):
#     # Convertir el frame a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Aplicar detección de bordes
#     edges = cv2.Canny(gray, 50, 150)
#     # Encontrar los contornos
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Buscar el contorno más grande que podría ser la caja
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         return x, y, w, h
#     return None

# # Función para detectar el ratón
# def detectar_raton(frame, roi):
#     x, y, w, h = roi
#     # Extraer la zona de interés (ROI)
#     roi_frame = frame[y:y + h, x:x + w]

#     # Convertir el frame a escala de grises
#     gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
#     # Aplicar un umbral para binarizar la imagen
#     _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
#     # Encontrar los contornos
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         # Filtrar contornos por tamaño para evitar detectar objetos grandes o pequeños no deseados
#         min_size, max_size = 500, 2000  # Ajusta según el tamaño del ratón
#         valid_contours = [cnt for cnt in contours if min_size < cv2.contourArea(cnt) < max_size]
#         if valid_contours:
#             # Encontrar el contorno más grande entre los válidos
#             largest_contour = max(valid_contours, key=cv2.contourArea)
#             # Calcular el centro del contorno
#             M = cv2.moments(largest_contour)
#             if M['m00'] != 0:
#                 cx = int(M['m10'] / M['m00']) + x
#                 cy = int(M['m01'] / M['m00']) + y
#                 # Ajustar el contorno a la posición de la ROI
#                 largest_contour = largest_contour + np.array([x, y])
#                 return (cx, cy), largest_contour
#     return None, None

# # Detectar la caja en el primer frame
# ret, frame = cap.read()
# if ret:
#     frame = resize_frame(frame, screen_width, screen_height)
#     roi = detectar_caja(frame)
#     if roi:
#         print(f"Caja detectada en: {roi}")
#     else:
#         print("No se pudo detectar la caja")
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
# else:
#     print("Error al leer el primer frame")
#     cap.release()
#     cv2.destroyAllWindows()
#     exit()

# # Procesar el resto del video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Redimensionar el frame para mejor visualización
#     frame = resize_frame(frame, screen_width, screen_height)

#     # Detectar el ratón
#     position, largest_contour = detectar_raton(frame, roi)
#     if position:
#         cx, cy = position
#         positions.append(position)
#         # Dibujar el contorno y el centro en el frame
#         cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

#     # Mostrar el frame con la detección
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # Extraer las posiciones en x e y
# x_positions = [pos[0] for pos in positions]
# y_positions = [pos[1] for pos in positions]

# # Graficar la trayectoria
# plt.plot(x_positions, y_positions, linestyle='-', linewidth=1.5, color='red')
# plt.title('Trayectoria del Ratón')
# plt.xlabel('Posición X')
# plt.ylabel('Posición Y')
# plt.gca().invert_yaxis()  # Invertir el eje Y para que coincida con la representación de la imagen
# plt.show()

#####################################################

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Cargar el video
# video_path = 'D:/Leonel/ISI/Tesis/OneDrive_2024-05-23/NOR hembras escopolamina completo/G2/Reconocimiento_2_G.mp4'
# cap = cv2.VideoCapture(video_path)

# # Verificar si el video se abrió correctamente
# if not cap.isOpened():
#     print("Error al abrir el video")
# else:
#     print("Video abierto exitosamente")

# positions = []

# # Obtener las dimensiones de la pantalla
# screen_width = 800
# screen_height = 600

# # Función para redimensionar el frame
# def resize_frame(frame, max_width, max_height):
#     (h, w) = frame.shape[:2]
#     if w > max_width or h > max_height:
#         # Calcular la proporción de redimensionamiento
#         ratio = min(max_width / float(w), max_height / float(h))
#         dim = (int(w * ratio), int(h * ratio))
#         resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#     else:
#         resized = frame
#     return resized

# # Función para detectar la caja en el primer frame
# def detectar_caja(frame):
#     # Convertir el frame a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Aplicar detección de bordes
#     edges = cv2.Canny(gray, 50, 150)
#     # Encontrar los contornos
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Buscar el contorno más grande que podría ser la caja
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         return x, y, w, h
#     return None

# # Función para detectar el ratón
# def detectar_raton(frame, roi):
#     x, y, w, h = roi
#     # Extraer la zona de interés (ROI)
#     roi_frame = frame[y:y + h, x:x + w]

#     # Convertir el frame a escala de grises
#     gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
#     # Aplicar un umbral para binarizar la imagen
#     _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
#     # Encontrar los contornos
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         # Filtrar contornos por tamaño para evitar detectar objetos grandes o pequeños no deseados
#         min_size, max_size = 500, 2000  # Ajusta según el tamaño del ratón
#         valid_contours = [cnt for cnt in contours if min_size < cv2.contourArea(cnt) < max_size]
#         if valid_contours:
#             # Encontrar el contorno más grande entre los válidos
#             largest_contour = max(valid_contours, key=cv2.contourArea)
#             # Calcular el centro del contorno
#             M = cv2.moments(largest_contour)
#             if M['m00'] != 0:
#                 cx = int(M['m10'] / M['m00']) + x
#                 cy = int(M['m01'] / M['m00']) + y
#                 # Ajustar el contorno a la posición de la ROI
#                 largest_contour = largest_contour + np.array([x, y])
#                 return (cx, cy), largest_contour
#     return None, None

# # Detectar la caja en el primer frame
# ret, frame = cap.read()
# if ret:
#     frame = resize_frame(frame, screen_width, screen_height)
#     roi = detectar_caja(frame)
#     if roi:
#         print(f"Caja detectada en: {roi}")
#     else:
#         print("No se pudo detectar la caja")
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
# else:
#     print("Error al leer el primer frame")
#     cap.release()
#     cv2.destroyAllWindows()
#     exit()

# # Procesar el resto del video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Redimensionar el frame para mejor visualización
#     frame = resize_frame(frame, screen_width, screen_height)

#     # Detectar el ratón
#     position, largest_contour = detectar_raton(frame, roi)
#     if position:
#         cx, cy = position
#         positions.append(position)
#         # Dibujar el contorno y el centro en el frame
#         cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

#     # Mostrar el frame con la detección
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # Extraer las posiciones en x e y
# x_positions = [pos[0] for pos in positions]
# y_positions = [pos[1] for pos in positions]

# # Graficar la trayectoria
# plt.plot(x_positions, y_positions, linestyle='-', linewidth=1.5, color='red')
# plt.title('Trayectoria del Ratón')
# plt.xlabel('Posición X')
# plt.ylabel('Posición Y')
# plt.gca().invert_yaxis()  # Invertir el eje Y para que coincida con la representación de la imagen
# plt.show()


###################################################################### SE DETECTA BASE DE LA CAJA #################

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Cargar el video
# video_path = 'D:/Leonel/ISI/Tesis/OneDrive_2024-05-23/NOR hembras escopolamina completo/G2/Reconocimiento_2_G.mp4'
# cap = cv2.VideoCapture(video_path)

# # Verificar si el video se abrió correctamente
# if not cap.isOpened():
#     print("Error al abrir el video")
# else:
#     print("Video abierto exitosamente")

# positions_mouse = []
# positions_objects = []

# # Obtener las dimensiones de la pantalla
# screen_width = 800
# screen_height = 600

# # Función para redimensionar el frame
# def resize_frame(frame, max_width, max_height):
#     (h, w) = frame.shape[:2]
#     if w > max_width or h > max_height:
#         # Calcular la proporción de redimensionamiento
#         ratio = min(max_width / float(w), max_height / float(h))
#         dim = (int(w * ratio), int(h * ratio))
#         resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#     else:
#         resized = frame
#     return resized

# # Función para detectar la caja en el primer frame
# def detectar_caja(frame):
#     # Convertir el frame a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Aplicar detección de bordes
#     edges = cv2.Canny(gray, 50, 150)
#     # Encontrar los contornos
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Buscar el contorno más grande que podría ser la caja
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         return x, y, w, h
#     return None

# # Función para detectar el ratón y los objetos adicionales
# def detectar_raton_y_objetos(frame, roi):
#     x, y, w, h = roi
#     # Extraer la zona de interés (ROI)
#     roi_frame = frame[y:y + h, x:x + w]

#     # Convertir el frame a escala de grises
#     gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
#     # Aplicar un umbral para binarizar la imagen
#     _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
#     # Encontrar los contornos
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     mouse_position = None
#     mouse_contour = None
#     object_positions = []

#     if contours:
#         # Filtrar contornos por tamaño para evitar detectar objetos grandes o pequeños no deseados
#         min_size_mouse, max_size_mouse = 500, 2000  # Ajusta según el tamaño del ratón
#         min_size_object, max_size_object = 100, 1000  # Ajusta según el tamaño de los objetos adicionales
#         valid_mouse_contours = [cnt for cnt in contours if min_size_mouse < cv2.contourArea(cnt) < max_size_mouse]
#         valid_object_contours = [cnt for cnt in contours if min_size_object < cv2.contourArea(cnt) < max_size_object]

#         if valid_mouse_contours:
#             # Encontrar el contorno más grande entre los válidos para el ratón
#             mouse_contour = max(valid_mouse_contours, key=cv2.contourArea)
#             # Calcular el centro del contorno del ratón
#             M = cv2.moments(mouse_contour)
#             if M['m00'] != 0:
#                 cx = int(M['m10'] / M['m00']) + x
#                 cy = int(M['m01'] / M['m00']) + y
#                 mouse_position = (cx, cy)
#                 # Ajustar el contorno a la posición de la ROI
#                 mouse_contour = mouse_contour + np.array([x, y])

#         for contour in valid_object_contours:
#             M = cv2.moments(contour)
#             if M['m00'] != 0:
#                 cx = int(M['m10'] / M['m00']) + x
#                 cy = int(M['m01'] / M['m00']) + y
#                 object_positions.append((cx, cy))
#                 # Ajustar el contorno a la posición de la ROI
#                 contour = contour + np.array([x, y])
    
#     return mouse_position, mouse_contour, object_positions

# # Detectar la caja en el primer frame
# ret, frame = cap.read()
# if ret:
#     frame = resize_frame(frame, screen_width, screen_height)
#     roi = detectar_caja(frame)
#     if roi:
#         print(f"Caja detectada en: {roi}")
#     else:
#         print("No se pudo detectar la caja")
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
# else:
#     print("Error al leer el primer frame")
#     cap.release()
#     cv2.destroyAllWindows()
#     exit()

# # Procesar el resto del video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Redimensionar el frame para mejor visualización
#     frame = resize_frame(frame, screen_width, screen_height)

#     # Dibujar la caja detectada en el primer frame
#     x, y, w, h = roi
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#     # Detectar el ratón y los objetos adicionales
#     mouse_position, mouse_contour, object_positions = detectar_raton_y_objetos(frame, roi)
#     if mouse_position:
#         cx, cy = mouse_position
#         positions_mouse.append(mouse_position)
#         # Dibujar el contorno y el centro del ratón en el frame
#         cv2.drawContours(frame, [mouse_contour], -1, (0, 255, 0), 2)
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

#     for obj_pos in object_positions:
#         cx, cy = obj_pos
#         positions_objects.append(obj_pos)
#         # Dibujar el centro del objeto en el frame
#         cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)  # Amarillo para los objetos adicionales

#     # Mostrar el frame con la detección
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # Extraer las posiciones en x e y
# x_positions_mouse = [pos[0] for pos in positions_mouse]
# y_positions_mouse = [pos[1] for pos in positions_mouse]
# x_positions_objects = [pos[0] for pos in positions_objects]
# y_positions_objects = [pos[1] for pos in positions_objects]

# # Graficar la trayectoria
# plt.plot(x_positions_mouse, y_positions_mouse, linestyle='-', linewidth=1.5, color='red', label='Ratón')
# plt.scatter(x_positions_objects, y_positions_objects, color='yellow', label='Objetos')
# plt.title('Trayectoria del Ratón y Objetos')
# plt.xlabel('Posición X')
# plt.ylabel('Posición Y')
# plt.gca().invert_yaxis()  # Invertir el eje Y para que coincida con la representación de la imagen
# plt.legend()
# plt.show()

##################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar el video
video_path = 'D:/Leonel/ISI/Tesis/OneDrive_2024-05-23/NOR hembras escopolamina completo/G2/Reconocimiento_2_G.mp4'
cap = cv2.VideoCapture(video_path)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el video")
else:
    print("Video abierto exitosamente")

positions_mouse = []
positions_objects = []


# Inicializar la distancia total recorrida
total_distance_cm = 0
# Suponiendo que la caja tiene dimensiones de 30 cm x 30 cm
box_width_cm = 30
box_height_cm = 30

# Obtener las dimensiones de la pantalla
screen_width = 800
screen_height = 600

# Función para redimensionar el frame
def resize_frame(frame, max_width, max_height):
    (h, w) = frame.shape[:2]
    if w > max_width or h > max_height:
        # Calcular la proporción de redimensionamiento
        ratio = min(max_width / float(w), max_height / float(h))
        dim = (int(w * ratio), int(h * ratio))
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    else:
        resized = frame
    return resized

# Función para detectar la caja en el primer frame
def detectar_caja(frame):
    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar detección de bordes
    edges = cv2.Canny(gray, 50, 150)
    # Encontrar los contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Buscar el contorno más grande que podría ser la caja
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    return None

# Función para detectar el ratón y los objetos adicionales
def detectar_raton_y_objetos(frame, roi):
    x, y, w, h = roi
    # Extraer la zona de interés (ROI)
    roi_frame = frame[y:y + h, x:x + w]

    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    # Aplicar un umbral para binarizar la imagen
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    # Encontrar los contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mouse_position = None
    mouse_contour = None
    object_positions = []

    if contours:
        # Filtrar contornos por tamaño para evitar detectar objetos grandes o pequeños no deseados
        min_size_mouse, max_size_mouse = 500, 1000  # Ajusta según el tamaño del ratón
        min_size_object, max_size_object = 60, 500 # Ajusta según el tamaño de los objetos adicionales
        valid_mouse_contours = [cnt for cnt in contours if min_size_mouse < cv2.contourArea(cnt) < max_size_mouse]
        valid_object_contours = [cnt for cnt in contours if min_size_object < cv2.contourArea(cnt) < max_size_object]

        if valid_mouse_contours:
            # Encontrar el contorno más grande entre los válidos para el ratón
            mouse_contour = max(valid_mouse_contours, key=cv2.contourArea)
            # Calcular el centro del contorno del ratón
            M = cv2.moments(mouse_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) + x
                cy = int(M['m01'] / M['m00']) + y
                mouse_position = (cx, cy)
                # Ajustar el contorno a la posición de la ROI
                mouse_contour = mouse_contour + np.array([x, y])

        for contour in valid_object_contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00']) + x
                cy = int(M['m01'] / M['m00']) + y
                object_positions.append((cx, cy))
                # Ajustar el contorno a la posición de la ROI
                contour = contour + np.array([x, y])
    
    return mouse_position, mouse_contour, object_positions

# Detectar la caja en el primer frame
ret, frame = cap.read()
if ret:
    frame = resize_frame(frame, screen_width, screen_height)
    roi = detectar_caja(frame)
    if roi:
        print(f"Caja detectada en: {roi}")
        roi_width_px = roi[2]
        roi_height_px = roi[3]
        # Escala de conversión de píxeles a centímetros
        scale_x = box_width_cm / roi_width_px
        scale_y = box_height_cm / roi_height_px
    else:
        print("No se pudo detectar la caja")
        cap.release()
        cv2.destroyAllWindows()
        exit()
else:
    print("Error al leer el primer frame")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Procesar el resto del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar el frame para mejor visualización
    frame = resize_frame(frame, screen_width, screen_height)

    # Dibujar la caja detectada en el primer frame
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Detectar el ratón y los objetos adicionales
    mouse_position, mouse_contour, object_positions = detectar_raton_y_objetos(frame, roi)
    if mouse_position:
        cx, cy = mouse_position
        positions_mouse.append(mouse_position)
        # Dibujar el contorno y el centro del ratón en el frame
        cv2.drawContours(frame, [mouse_contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Calcular la distancia euclidiana entre el ratón actual y el anterior
        if len(positions_mouse) > 1:
            prev_x, prev_y = positions_mouse[-2]
            distance_px = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
            distance_cm = distance_px * scale_x  # O scale_y, ya que son iguales
            total_distance_cm += distance_cm

    for obj_pos in object_positions:
        cx, cy = obj_pos
        positions_objects.append(obj_pos)
        # Dibujar el centro del objeto en el frame
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Amarillo para los objetos adicionales

    # Mostrar el frame con la detección
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Extraer las posiciones en x e y
x_positions_mouse = [pos[0] for pos in positions_mouse]
y_positions_mouse = [pos[1] for pos in positions_mouse]
x_positions_objects = [pos[0] for pos in positions_objects]
y_positions_objects = [pos[1] for pos in positions_objects]

# Graficar la trayectoria
plt.plot(x_positions_mouse, y_positions_mouse, linestyle='-', linewidth=1.5, color='red', label='Ratón')
plt.scatter(x_positions_objects, y_positions_objects, color='yellow', label='Objetos')
plt.title('Trayectoria del Ratón y Objetos')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')
plt.gca().invert_yaxis()  # Invertir el eje Y para que coincida con la representación de la imagen
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)  # Colocar la leyenda fuera de la gráfica
plt.subplots_adjust(right=0.75)  # Ajustar para que haya espacio para la leyenda
plt.show()

print("Distancia total recorrida por el ratón:", total_distance_cm, "cm")

##########################################################################

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Cargar el video
# video_path = 'D:/Leonel/ISI/Tesis/OneDrive_2024-05-23/NOR hembras escopolamina completo/G2/Reconocimiento_2_G.mp4'
# cap = cv2.VideoCapture(video_path)

# # Verificar si el video se abrió correctamente
# if not cap.isOpened():
#     print("Error al abrir el video")
# else:
#     print("Video abierto exitosamente")

# positions_mouse = []
# positions_rectangles = []
# positions_circles = []

# # Obtener las dimensiones de la pantalla
# screen_width = 800
# screen_height = 600

# # Función para redimensionar el frame
# def resize_frame(frame, max_width, max_height):
#     (h, w) = frame.shape[:2]
#     if w > max_width or h > max_height:
#         # Calcular la proporción de redimensionamiento
#         ratio = min(max_width / float(w), max_height / float(h))
#         dim = (int(w * ratio), int(h * ratio))
#         resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#     else:
#         resized = frame
#     return resized

# # Función para detectar la caja en el primer frame
# def detectar_caja(frame):
#     # Convertir el frame a escala de grises
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Aplicar detección de bordes
#     edges = cv2.Canny(gray, 50, 150)
#     # Encontrar los contornos
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # Buscar el contorno más grande que podría ser la caja
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         return x, y, w, h
#     return None

# # Función para detectar el ratón y los objetos adicionales
# def detectar_raton_y_objetos(frame, roi):
#     x, y, w, h = roi
#     # Extraer la zona de interés (ROI)
#     roi_frame = frame[y:y + h, x:x + w]

#     # Convertir el frame a escala de grises
#     gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
#     # Aplicar un umbral para binarizar la imagen
#     _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

#     # Aplicar apertura y cierre para limpiar la imagen binaria
#     kernel = np.ones((3, 3), np.uint8)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#     # Encontrar los contornos
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     mouse_position = None
#     mouse_contour = None
#     rectangle_positions = []
#     circle_positions = []

#     if contours:
#         # Filtrar contornos por tamaño para evitar detectar objetos grandes o pequeños no deseados
#         min_size_mouse, max_size_mouse = 300, 1500  # Ajusta según el tamaño del ratón
#         min_size_object, max_size_object = 100, 1000  # Ajusta según el tamaño de los objetos adicionales
#         valid_mouse_contours = [cnt for cnt in contours if min_size_mouse < cv2.contourArea(cnt) < max_size_mouse]
#         valid_object_contours = [cnt for cnt in contours if min_size_object < cv2.contourArea(cnt) < max_size_object]

#         if valid_mouse_contours:
#             # Encontrar el contorno más grande entre los válidos para el ratón
#             mouse_contour = max(valid_mouse_contours, key=cv2.contourArea)
#             # Calcular el centro del contorno del ratón
#             M = cv2.moments(mouse_contour)
#             if M['m00'] != 0:
#                 cx = int(M['m10'] / M['m00']) + x
#                 cy = int(M['m01'] / M['m00']) + y
#                 mouse_position = (cx, cy)
#                 # Ajustar el contorno a la posición de la ROI
#                 mouse_contour = mouse_contour + np.array([x, y])

#         # Detectar rectángulos y círculos
#         for cnt in valid_object_contours:
#             approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
#             if len(approx) == 4:  # Rectángulo
#                 M = cv2.moments(cnt)
#                 if M['m00'] != 0:
#                     cx = int(M['m10'] / M['m00']) + x
#                     cy = int(M['m01'] / M['m00']) + y
#                     rectangle_positions.append((cx, cy))
#                     # Ajustar el contorno a la posición de la ROI
#                     cnt = cnt + np.array([x, y])
#             elif len(approx) > 4:  # Círculo (o algo similar)
#                 (cx, cy), radius = cv2.minEnclosingCircle(cnt)
#                 if radius > 10:  # Ajustar el tamaño mínimo del círculo
#                     circle_positions.append((int(cx) + x, int(cy) + y))

#     return mouse_position, mouse_contour, rectangle_positions, circle_positions

# # Detectar la caja en el primer frame
# ret, frame = cap.read()
# if ret:
#     frame = resize_frame(frame, screen_width, screen_height)
#     roi = detectar_caja(frame)
#     if roi:
#         print(f"Caja detectada en: {roi}")
#     else:
#         print("No se pudo detectar la caja")
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
# else:
#     print("Error al leer el primer frame")
#     cap.release()
#     cv2.destroyAllWindows()
#     exit()

# # Procesar el resto del video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Redimensionar el frame para mejor visualización
#     frame = resize_frame(frame, screen_width, screen_height)

#     # Dibujar la caja detectada en el primer frame
#     x, y, w, h = roi
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#     # Detectar el ratón y los objetos adicionales
#     mouse_position, mouse_contour, rectangle_positions, circle_positions = detectar_raton_y_objetos(frame, roi)
#     if mouse_position:
#         cx, cy = mouse_position
#         positions_mouse.append(mouse_position)
#         # Dibujar el contorno y el centro del ratón en el frame
#         cv2.drawContours(frame, [mouse_contour], -1, (0, 255, 0), 2)
#         cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

#     for rect_pos in rectangle_positions:
#         cx, cy = rect_pos
#         positions_rectangles.append(rect_pos)
#         # Dibujar el centro del rectángulo en el frame
#         cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)  # Amarillo para los rectángulos

#     for circ_pos in circle_positions:
#         cx, cy = circ_pos
#         positions_circles.append(circ_pos)
#         # Dibujar el centro del círculo en el frame
#         cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Cyan para los círculos

#     # Mostrar el frame con la detección
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # Extraer las posiciones en x e y
# x_positions_mouse = [pos[0] for pos in positions_mouse]
# y_positions_mouse = [pos[1] for pos in positions_mouse]
# x_positions_rectangles = [pos[0] for pos in positions_rectangles]
# y_positions_rectangles = [pos[1] for pos in positions_rectangles]
# x_positions_circles = [pos[0] for pos in positions_circles]
# y_positions_circles = [pos[1] for pos in positions_circles]

# # Graficar la trayectoria
# plt.figure(figsize=(10, 8))
# plt.plot(x_positions_mouse, y_positions_mouse, linestyle='-', linewidth=1.5, color='red', label='Ratón')
# plt.scatter(x_positions_rectangles, y_positions_rectangles, color='yellow', label='Rectángulos')
# plt.scatter(x_positions_circles, y_positions_circles, color='cyan', label='Círculos')
# plt.title('Trayectoria del Ratón y Objetos Detectados')
# plt.xlabel('Posición X')
# plt.ylabel('Posición Y')
# plt.gca().invert_yaxis()  # Invertir el eje Y para que coincida con la representación de la imagen
# plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.subplots_adjust(right=0.75)  # Ajustar para que haya espacio para la leyenda
# plt.show()