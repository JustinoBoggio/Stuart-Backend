# import sys
# import os
# import cv2
# import torch
# import yaml
# import numpy as np
# from tqdm import tqdm  # Importar tqdm para la barra de progreso
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as MplPolygon
# import matplotlib.patches as patches
# from shapely.geometry import LineString, Polygon, Point
# from collections import defaultdict

# # Añadir el directorio padre al sys.path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

# # Importar funciones de inferencia
# from models.keypoint_detection.infer_keypoints import load_keypoint_model, get_keypoints, draw_keypoints
# from models.yolov11_segmentation.infer_yolo import load_yolo_model, get_yolo_detections, draw_yolo_detections

# def expand_mask_shapely(mask, escala, ampliacion_cm=10):
#     """
#     Expande la máscara añadiendo una distancia específica en centímetros a cada lado utilizando Shapely.

#     Args:
#         mask (torch.Tensor): Máscara original.
#         escala (float): Escala en píxeles por centímetro (px/cm).
#         ampliacion_cm (float): Distancia a añadir en centímetros.

#     Returns:
#         list: Lista de puntos del polígono expandido.
#     """
#     # Convertir la máscara a NumPy
#     mask_numpy = mask.cpu().numpy()
#     if mask_numpy.ndim == 3 and mask_numpy.shape[0] == 1:
#         mask_numpy = mask_numpy.squeeze(0)  # De [1, H, W] a [H, W]
#     mask_binary = (mask_numpy * 255).astype(np.uint8)

#     # Encontrar contornos en la máscara
#     contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         print("Advertencia: No se encontraron contornos en la máscara.")
#         return []

#     # Asumir que el contorno más grande es el objeto actual
#     largest_contour = max(contours, key=cv2.contourArea)

#     # Convertir el contorno a una lista de puntos
#     points = [tuple(point[0]) for point in largest_contour]

#     # Crear un polígono de Shapely
#     polygon = Polygon(points)
#     if not polygon.is_valid:
#         polygon = polygon.buffer(0)  # Corregir si es inválido

#     # Ampliar el polígono
#     buffer_distance = ampliacion_cm * escala  # Convertir cm a píxeles
#     polygon_expanded = polygon.buffer(buffer_distance)

#     if not polygon_expanded.is_valid:
#         print("Advertencia: Polígono expandido inválido.")
#         return []

#     # Obtener los puntos del exterior del polígono expandido
#     exterior_coords = np.array(polygon_expanded.exterior.coords).astype(int).tolist()

#     return exterior_coords

# # Definir las funciones de padding y redimensión
# def pad_to_square(image, desired_size=1920, color=[0, 0, 0]):
#     """
#     Añade padding a la imagen para hacerla cuadrada, adaptando el tamaño si se especifica.

#     Args:
#         image (numpy.ndarray): Imagen original en formato BGR.
#         desired_size (int): Tamaño deseado para el lado cuadrado.
#         color (list): Color del padding en formato BGR.

#     Returns:
#         numpy.ndarray: Imagen cuadrada con padding.
#     """
#     old_size = image.shape[:2]  # (altura, ancho)
#     delta_w = desired_size - old_size[1]
#     delta_h = desired_size - old_size[0]
#     top, bottom = delta_h // 2, delta_h - (delta_h // 2)
#     left, right = delta_w // 2, delta_w - (delta_w // 2)
#     new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
#     return new_image

# def resize_image(image, size=(640, 640)):
#     """
#     Redimensiona la imagen a un tamaño específico.

#     Args:
#         image (numpy.ndarray): Imagen en formato BGR.
#         size (tuple): Nueva resolución (ancho, alto).

#     Returns:
#         numpy.ndarray: Imagen redimensionada.
#     """
#     resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
#     return resized

# def pad_and_resize(image, desired_size=1920, final_size=(640, 640), color=[0, 0, 0]):
#     """
#     Aplica padding para hacer la imagen cuadrada y luego la redimensiona a la resolución final.

#     Args:
#         image (numpy.ndarray): Imagen original.
#         desired_size (int): Tamaño deseado para el lado cuadrado.
#         final_size (tuple): Resolución final (ancho, alto).
#         color (list): Color del padding en formato BGR.

#     Returns:
#         numpy.ndarray: Imagen redimensionada a la resolución final con padding aplicado.
#     """
#     padded_image = pad_to_square(image, desired_size, color)
#     resized_image = resize_image(padded_image, final_size)
#     return resized_image

# def evaluate_keypoints(keypoints, image_shape=(640, 640)):
#     """
#     Evalúa la validez de los keypoints detectados.
#     Por ejemplo, verifica si las coordenadas están dentro del rango de la imagen.

#     Args:
#         keypoints (list): Lista de keypoints detectados.
#         image_shape (tuple): Tamaño de la imagen (alto, ancho).
#     """
#     for kp in keypoints:
#         x, y = kp['position']
#         if not (0 <= x < image_shape[1] and 0 <= y < image_shape[0]):
#             print(f"Keypoint {kp['name']} fuera de los límites: ({x}, {y})")


# def calculate_bounds(keypoint_trajectories, current_objects, zona_botellas, zona_rastis, padding=50):
#     """
#     Calcula los límites mínimos y máximos de las coordenadas para ajustar el zoom.

#     Args:
#         keypoint_trajectories (dict): Trayectorias de keypoints.
#         current_objects (defaultdict): Objetos detectados con sus contornos.
#         zona_botellas (dict): Zonas ampliadas de las Botellas.
#         zona_rastis (dict): Zonas ampliadas de los Rastis.
#         padding (int): Espacio adicional alrededor de los puntos para el zoom.

#     Returns:
#         (xmin, xmax, ymin, ymax)
#     """
#     all_x = []
#     all_y = []

#     # Keypoints
#     for positions in keypoint_trajectories.values():
#         for pos in positions:
#             all_x.append(pos[0])
#             all_y.append(pos[1])

#     # Objetos
#     for contours in current_objects.values():
#         for contour in contours:
#             polygon = contour.reshape(-1, 2)
#             all_x.extend(polygon[:, 0])
#             all_y.extend(polygon[:, 1])

#     # Zonas Botellas
#     for points_dilated in zona_botellas.values():
#         points = np.array(points_dilated)
#         all_x.extend(points[:, 0])
#         all_y.extend(points[:, 1])

#     # Zonas Rastis
#     for points_dilated in zona_rastis.values():
#         points = np.array(points_dilated)
#         all_x.extend(points[:, 0])
#         all_y.extend(points[:, 1])

#     xmin, xmax = min(all_x) - padding, max(all_x) + padding
#     ymin, ymax = min(all_y) - padding, max(all_y) + padding

#     # Asegurar que los límites no salgan de la imagen
#     xmin = max(xmin, 0)
#     ymin = max(ymin, 0)
#     xmax = min(xmax, 640)
#     ymax = min(ymax, 640)

#     return xmin, xmax, ymin, ymax

# def post_processing(keypoint_trajectories,
#                     distance_traveled,
#                     tiempo_zonas,
#                     trajectories_dir,
#                     video_name,
#                     current_objects, 
#                     zona_botellas, 
#                     zona_rastis,
#                     class_colors):
#     """
#     Genera mapas de trayectoria que incluyen la base de la caja, los objetos y sus zonas ampliadas.
    
#     Args:
#         keypoint_trajectories (dict): Trayectorias de keypoints.
#         distance_traveled (dict): Distancia recorrida por keypoints.
#         tiempo_zonas (dict): Tiempo pasado en zonas de objetos.
#         trajectories_dir (str): Directorio para guardar los resultados.
#         video_name (str): Nombre del video procesado.
#         current_objects (defaultdict): Objetos detectados con sus contornos.
#         zona_botellas (dict): Zonas ampliadas de las Botellas.
#         zona_rastis (dict): Zonas ampliadas de los Rastis.
#     """
#     # Calcular los límites para el zoom
#     xmin, xmax, ymin, ymax = calculate_bounds(keypoint_trajectories, current_objects, zona_botellas, zona_rastis)

#     # Dibujar las trayectorias de keypoints
#     for kp_name, positions in keypoint_trajectories.items():
#         # Crear una figura con fondo blanco
#         fig, ax = plt.subplots(figsize=(8, 8))
#         ax.set_facecolor('white')  # Fondo blanco
#         # Mantener la proporción de aspecto
#         ax.set_aspect('equal')

#         # # Establecer límites del gráfico
#         # ax.set_xlim(0, 640)
#         # ax.set_ylim(640, 0)  # Invertir el eje Y para que coincida con la imagen

#         # Establecer límites del gráfico con zoom
#         ax.set_xlim(xmin, xmax)
#         ax.set_ylim(ymax, ymin)  # Invertir el eje Y para que coincida con la imagen

#         # Ocultar los ejes
#         ax.axis('off')

#         x = [pos[0] for pos in positions]
#         y = [pos[1] for pos in positions]
#         ax.plot(x, y, marker='o', markersize=2, label=f'Trayectoria {kp_name}')

#         # Añadir etiqueta de distancia
#         distancia = distance_traveled.get(kp_name, 0.0)
#         # ax.text(10, 30 + 15 * list(keypoint_trajectories.keys()).index(kp_name), 
#         #         f"Distancia {kp_name}: {distancia:.2f} m", 
#         #         color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
#         ax.text(xmin + 10, ymin + 30, 
#                 f"Distancia {kp_name}: {distancia:.2f} m", 
#                 color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

#         # Dibujar los objetos detectados
#         class_labels = {'BaseCaja': 'BaseCaja', 'Botella': 'Botella', 'Rasti': 'Rasti'}
#         class_colors_plot = {'BaseCaja': 'gray', 'Botella': 'orange', 'Rasti': 'blue'}

#         # Usar un diccionario para mantener las leyendas únicas
#         legend_handles = {}

#         for class_name, contours in current_objects.items():
#             for contour in contours:
#                 # Convertir el contorno a una forma adecuada para matplotlib
#                 contour = contour.reshape(-1, 2)
                
#                 # Asegurar que el polígono esté cerrado
#                 if not np.array_equal(contour[0], contour[-1]):
#                     contour = np.vstack([contour, contour[0]])
                
#                 # Crear un polígono y añadirlo al plot
#                 polygon = MplPolygon(contour, closed=True, fill=None, edgecolor=class_colors_plot[class_name], linewidth=2, label=class_labels[class_name])
#                 ax.add_patch(polygon)

#                 # Añadir al diccionario de handles para leyenda (solo una vez por clase)
#                 if class_name not in legend_handles:
#                     legend_handles[class_name] = patches.Patch(edgecolor=class_colors_plot[class_name], facecolor='none', label=class_labels[class_name])

#         # Dibujar zonas extendidas (Botellas y Rastis)
#         for zona_id, points_dilated in zona_botellas.items():
#             points = np.array(points_dilated)
            
#             # Asegurar que el polígono esté cerrado
#             if not np.array_equal(points[0], points[-1]):
#                 points = np.vstack([points, points[0]])
            
#             polygon = MplPolygon(points, closed=True, fill=None, edgecolor='orange', linestyle='--', linewidth=1, label='Zona Botella')
#             ax.add_patch(polygon)
#             # Añadir al diccionario de handles para leyenda
#             if 'Zona Botella' not in legend_handles:
#                 legend_handles['Zona Botella'] = patches.Patch(edgecolor='orange', facecolor='none', linestyle='--', label='Zona Botella')

#         for zona_id, points_dilated in zona_rastis.items():
#             points = np.array(points_dilated)
            
#             # Asegurar que el polígono esté cerrado
#             if not np.array_equal(points[0], points[-1]):
#                 points = np.vstack([points, points[0]])
            
#             polygon = MplPolygon(points, closed=True, fill=None, edgecolor='blue', linestyle='--', linewidth=1, label='Zona Rasti')
#             ax.add_patch(polygon)
#             # Añadir al diccionario de handles para leyenda
#             if 'Zona Rasti' not in legend_handles:
#                 legend_handles['Zona Rasti'] = patches.Patch(edgecolor='blue', facecolor='none', linestyle='--', label='Zona Rasti')

#         # Añadir leyenda
#         #ax.legend(handles=legend_handles.values(), loc='upper right')

#         # Añadir leyenda fuera del gráfico
#         ax.legend(handles=legend_handles.values(), loc='upper right', bbox_to_anchor=(1.10, 1.05), borderaxespad=0.)

#         # Título y etiquetas
#         ax.set_title(f'Mapa de Trayectoria {kp_name} - {video_name}')
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')

#         # Guardar la figura
#         plt.savefig(os.path.join(trajectories_dir, f'Trayectoria_{kp_name}_con_distancia.png'))
#         plt.close()

#     # Guardar distancias recorridas
#     with open(os.path.join(trajectories_dir, 'distancias_recorridas.txt'), 'w') as f:
#         for kp_name, distancia in distance_traveled.items():
#             f.write(f"Keypoint {kp_name}: {distancia:.4f} metros\n")

#     # Guardar tiempos en zonas
#     with open(os.path.join(trajectories_dir, 'tiempos_zonas.txt'), 'w') as f:
#         total_time = 0.0
#         for obj_id, tiempo in tiempo_zonas.items():
#             f.write(f"Objeto {obj_id}: {tiempo:.4f} segundos\n")
#             total_time += tiempo
#         f.write(f"Tiempo total en zonas: {total_time:.4f} segundos\n")



# def main(video_path, keypoint_config, keypoint_model_path, yolo_model_path):
#     # Crear directorio de salida para mapas de trayectoria
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     trajectories_dir = os.path.join("outputs", f"processed_{video_name}")
#     os.makedirs(trajectories_dir, exist_ok=True)

#     output_path = f"outputs/processed_{video_name}/{video_name}_processed.mp4"  # Ruta al video de salida (opcional)
#     # Asegúrate de que el directorio de salida existe
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     # Variables para almacenar datos
#     keypoint_trajectories = {}  # Diccionario para almacenar trayectorias de keypoints
#     distance_traveled = {}       # Diccionario para almacenar distancia recorrida
#     tiempo_zonas = {}            # Diccionario para almacenar tiempo en zonas de objetos
#     #current_objects = []         # Lista para almacenar objetos detectados en el primer frame
#     current_objects = defaultdict(list)  # Usar defaultdict para listas
#     zona_botellas = {}           # Diccionario para almacenar zonas de Botellas
#     zona_rastis = {}             # Diccionario para almacenar zonas de Rastis
#     escala = 0.0                 # Escala: px/cm, se calculará después de la detección inicial
    
#     # Cargar modelos
#     print("Cargando modelo de detección de keypoints...")
#     keypoint_model, keypoint_device = load_keypoint_model(keypoint_config, keypoint_model_path)
#     print("Modelo de keypoints cargado.")
    
#     print("Cargando modelo de segmentación YOLOv11...")
#     yolo_model = load_yolo_model(yolo_model_path)
#     print("Modelo YOLOv11 cargado.")
    
#     # Definir colores para YOLO en formato BGR
#     class_colors = {
#         'BaseCaja': (128, 128, 128),  # Gris
#         'Botella': (0, 165, 255),     # Naranja
#         'Rasti': (255, 0, 0),         # Azul
#         # Añade más clases si es necesario
#     }

#     class_colors_matplotlib = {
#     'BaseCaja': 'gray',
#     'Botella': 'orange',
#     'Rasti': 'blue',
# }
    
#     target_classes = ['BaseCaja', 'Botella', 'Rasti']
#     target_class_name = 'BaseCaja'

#     # Abrir video
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: No se pudo abrir el video {video_path}")
#         return
    
#     # Obtener propiedades del video
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     print(f"Resolución del video: {frame_width}x{frame_height} píxeles")
#     print(f"FPS del video: {fps}")
#     print(f"Número total de frames: {total_frames}")
    
#     # Definir el codec y crear VideoWriter si se desea guardar el output
#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Cambia el codec si es necesario
#         out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))
#         print(f"Video de salida guardado en: {output_path}")
#     else:
#         out = None
    
#     # Iniciar la barra de progreso
#     with tqdm(total=total_frames, desc="Procesando Video", unit="frame") as pbar:
#         frame_count = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
#             print(f"\nProcesando frame {frame_count}/{total_frames}")
            
#             # Aplicar padding y redimensión
#             processed_frame = pad_and_resize(frame, desired_size=1920, final_size=(640, 640), color=[0, 0, 0])
            
            
#             if frame_count == 1:
#                 # Realizar detección con YOLO solo en el primer frame
#                 yolo_results = get_yolo_detections(yolo_model, processed_frame)
#                 # current_objects_detection = yolo_results
#                 # # Dibujar los objetos detectados
#                 # final_frame = draw_yolo_detections(processed_frame, yolo_results, class_colors, target_class_name) #processed_frame
                
#                 print(f"Resultado: {yolo_results}")

#                 # Extraer detecciones de YOLO para procesar 'BaseCaja', 'Botella' y 'Rasti'
#                 detections = []
#                 for result in yolo_results:
#                     # result es un objeto de la clase Results
#                     masks = result.masks
#                     boxes = result.boxes
#                     if masks is None or boxes is None:
#                         continue
#                     for mask, box in zip(masks.data, boxes):
#                         class_id = int(box.cls.cpu().numpy())
#                         class_name = yolo_model.names[class_id]
#                         bbox = box.xyxy.cpu().numpy().astype(int).tolist()  # [x1, y1, x2, y2]
#                         detections.append({'class': class_name, 'bbox': bbox, 'mask': mask})
                
#                 # Identificar BaseCaja para calcular la escala
#                 base_caja = next((obj for obj in detections if obj['class'] == 'BaseCaja'), None)
#                 if base_caja:
#                     mask = base_caja['mask']  # La máscara correspondiente a BaseCaja
#                     try:
#                         # Convertir la máscara a un arreglo de NumPy
#                         mask_numpy = mask.cpu().numpy()
#                         if mask_numpy.ndim == 3 and mask_numpy.shape[0] == 1:
#                             mask_numpy = mask_numpy.squeeze(0)  # Convertir de [1, H, W] a [H, W]
#                         mask_binary = (mask_numpy * 255).astype(np.uint8)
                        
#                         # Guardar la máscara para depuración (opcional)
#                         cv2.imwrite(os.path.join(trajectories_dir, f"mask_base_caja_frame{frame_count}.png"), mask_binary)
                        
#                         # Encontrar contornos en la máscara
#                         contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                         if not contours:
#                             print("Error: No se encontraron contornos en la máscara de BaseCaja.")
#                             return
                        
#                         # Asumir que el contorno más grande es BaseCaja
#                         largest_contour = max(contours, key=cv2.contourArea)
                        
#                         # Aproximar el contorno a un polígono con menos puntos
#                         epsilon = 0.02 * cv2.arcLength(largest_contour, True)
#                         approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
#                         # Verificar que tenemos cuatro puntos
#                         if len(approx) != 4:
#                             print(f"Advertencia: Se encontraron {len(approx)} puntos en el contorno de BaseCaja, se esperaban 4.")
                        
#                         # Extraer los puntos
#                         points = [point[0] for point in approx]
#                         print(f"Puntos del contorno de BaseCaja: {points}")  # Depuración
                        
#                         #current_objects.append({'class': 'BaseCaja', 'approx': approx})
#                         #current_objects['BaseCaja'] = approx
#                         approx = approx.reshape(-1, 1, 2).astype(np.int32)
#                         current_objects['BaseCaja'].append(approx)

#                         # Calcular las distancias entre puntos consecutivos
#                         distances = []
#                         for i in range(len(points)):
#                             p1 = points[i]
#                             p2 = points[(i + 1) % len(points)]
#                             dx = p2[0] - p1[0]
#                             dy = p2[1] - p1[1]
#                             distance = np.sqrt(dx**2 + dy**2)
#                             distances.append(distance)
#                             print(f"Distancia entre {p1} y {p2}: {distance} px")  # Depuración
                        
#                         # Calcular la escala como el promedio de las distancias dividido por 50 cm
#                         average_distance_px = np.mean(distances)
#                         real_size_cm = 40  # Tamaño real de BaseCaja en cm
#                         escala = average_distance_px / real_size_cm  # px/cm
#                         print(f"Escala calculada: {escala:.2f} px/cm")
#                     except AttributeError:
#                         print("Error: No se pudo acceder al tensor de la máscara de BaseCaja.")
#                         return
#                 else:
#                     print("Error: No se detectó BaseCaja en el primer frame para calcular la escala.")
#                     return
                
#                 # Procesar las demás clases: Botella y Rasti
#                 for obj in detections:
#                     class_name = obj['class']
#                     if class_name not in ['Botella', 'Rasti']:
#                         continue  # Solo procesar Botella y Rasti
                    
#                     mask = obj['mask']
#                     try:
#                         # Expandir la máscara y obtener el polígono dilatado
#                         points_dilated = expand_mask_shapely(mask, escala, ampliacion_cm=3)
#                         if not points_dilated:
#                             continue  # Saltar si no se pudo expandir
                        
#                         print(f"Puntos del contorno dilatado de {class_name}: {points_dilated}")  # Depuración

#                         # Dibujar el polígono expandido en el frame
#                         cv2.polylines(processed_frame, [np.array(points_dilated)], isClosed=True, color=class_colors[class_name], thickness=2)
                        
#                         # Opcional: Almacenar las zonas ampliadas
#                         zona_id = f"{class_name}_{len(zona_botellas) + len(zona_rastis) + 1}"
#                         if class_name == 'Botella':
#                             zona_botellas[zona_id] = points_dilated
#                         elif class_name == 'Rasti':
#                             zona_rastis[zona_id] = points_dilated

#                         mask_numpy = mask.cpu().numpy()
#                         if mask_numpy.ndim == 3 and mask_numpy.shape[0] == 1:
#                             mask_numpy = mask_numpy.squeeze(0)  # Convertir de [1, H, W] a [H, W]
#                         mask_binary = (mask_numpy * 255).astype(np.uint8)

#                         # Encontrar contornos en la máscara
#                         contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                         if not contours:
#                             print(f"Error: No se encontraron contornos en la máscara de {class_name}.")
#                             return
                        
#                         largest_contour = max(contours, key=cv2.contourArea)

#                         # Aproximar el contorno a un polígono con menos puntos
#                         epsilon = 0.02 * cv2.arcLength(largest_contour, True)
#                         approx = cv2.approxPolyDP(largest_contour, epsilon, True)

#                         approx = approx.reshape(-1, 1, 2).astype(np.int32)
#                         #current_objects.append({'class': class_name, 'approx': approx})
#                         current_objects[class_name].append(approx)
#                         #current_objects.append[class_name] = approx

#                     except AttributeError:
#                         print(f"Error: No se pudo acceder al tensor de la máscara de {class_name}.")
#                         continue
                
#                 final_frame = processed_frame.copy()

#                 # Procesar con el modelo de keypoints
#                 keypoints = get_keypoints(keypoint_model, keypoint_device, processed_frame)
#                 evaluate_keypoints(keypoints)
#                 #print(f"Keypoints detectados: {keypoints}")  # Depuración
#                 draw_keypoints(final_frame, keypoints)

#                 #Dibujar BaseCaja, Botella, Rasti
#                 for class_name, approx in current_objects.items():
#                     cv2.polylines(final_frame, approx, isClosed=True, color=class_colors[class_name], thickness=2)

#             else:
#                 # Procesar con el modelo de keypoints
#                 keypoints = get_keypoints(keypoint_model, keypoint_device, processed_frame)
#                 evaluate_keypoints(keypoints)
#                 print(f"Keypoints detectados: {keypoints}")  # Depuración

#                 # Para frames posteriores, dibujar las zonas ampliadas almacenadas
#                 final_frame = processed_frame.copy()
#                 # Dibujar keypoints
#                 draw_keypoints(final_frame, keypoints)

#                 #Dibujar BaseCaja, Botella, Rasti
#                 for class_name, approx in current_objects.items():
#                     print(f"Class: {class_name} - Approx: {approx}")
#                     cv2.polylines(final_frame, approx, isClosed=True, color=class_colors[class_name], thickness=2)

#                 #Dibujar zonas expandidas directamente
#                 for obj_id, points_dilated in zona_botellas.items():
#                     cv2.polylines(final_frame, [np.array(points_dilated, dtype=np.int32)], isClosed=True, color=class_colors['Botella'], thickness=2)
#                 for obj_id, points_dilated in zona_rastis.items():
#                     cv2.polylines(final_frame, [np.array(points_dilated, dtype=np.int32)], isClosed=True, color=class_colors['Rasti'], thickness=2)
                
#                 # Dibujar zonas traslúcidas
#                 # alpha = 0.3  # Transparencia
#                 # overlay = final_frame.copy()
#                 # for obj_id, points_dilated in zona_botellas.items():
#                 #     cv2.polylines(overlay, [np.array(points_dilated, dtype=np.int32)], isClosed=True, color=class_colors['Botella'], thickness=2)
#                 # for obj_id, points_dilated in zona_rastis.items():
#                 #     cv2.polylines(overlay, [np.array(points_dilated, dtype=np.int32)], isClosed=True, color=class_colors['Rasti'], thickness=2)
#                 # cv2.addWeighted(overlay, alpha, final_frame, 1 - alpha, 0, final_frame)
            
#             # Actualizar trayectorias de keypoints
#             for kp in keypoints:
#                 name = kp['name']
#                 pos = kp['position']
#                 if name not in keypoint_trajectories:
#                     keypoint_trajectories[name] = []
#                 keypoint_trajectories[name].append(pos)
                
#                 # Calcular distancia recorrida
#                 if name not in distance_traveled:
#                     distance_traveled[name] = 0.0
#                 if len(keypoint_trajectories[name]) > 1:
#                     prev_pos = keypoint_trajectories[name][-2]
#                     dx = pos[0] - prev_pos[0]
#                     dy = pos[1] - prev_pos[1]
#                     distance = np.sqrt(dx**2 + dy**2) / escala / 100  # metros
#                     distance_traveled[name] += distance
            
#             # # Contabilizar el tiempo que la "Nariz" pasa por las zonas
#             # for kp in keypoints:
#             #     if kp['name'] == 'Nariz':
#             #         nariz_pos = kp['position']
#             #         for obj_id, points_dilated in zona_botellas.items():
#             #             # Convertir la lista de puntos a un array de numpy
#             #             polygon = np.array(points_dilated, dtype=np.int32)
#             #             # Verificar si la nariz está dentro del polígono
#             #             result = cv2.pointPolygonTest(polygon, tuple(nariz_pos), False)
#             #             if result >= 0:
#             #                 if obj_id not in tiempo_zonas:
#             #                     tiempo_zonas[obj_id] = 0.0
#             #                 tiempo_zonas[obj_id] += 1 / fps  # Incrementar tiempo en segundos
#             #         for obj_id, points_dilated in zona_rastis.items():
#             #             polygon = np.array(points_dilated, dtype=np.int32)
#             #             result = cv2.pointPolygonTest(polygon, tuple(nariz_pos), False)
#             #             if result >= 0:
#             #                 if obj_id not in tiempo_zonas:
#             #                     tiempo_zonas[obj_id] = 0.0
#             #                 tiempo_zonas[obj_id] += 1 / fps  # Incrementar tiempo en segundos

#             # Inicializar variables de posiciones de keypoints
#             nariz_pos = None
#             nuca_pos = None
#             oreja_izq_pos = None
#             oreja_der_pos = None
#             mitad_columna_pos = None
#             base_cola_pos = None

#             # Obtener posiciones de keypoints
#             for kp in keypoints:
#                 if kp['name'] == 'Nariz':
#                     nariz_pos = kp['position']
#                 elif kp['name'] == 'Nuca':
#                     nuca_pos = kp['position']
#                 elif kp['name'] == 'Oreja Izquierda':
#                     oreja_izq_pos = kp['position']
#                 elif kp['name'] == 'Oreja Derecha':
#                     oreja_der_pos = kp['position']
#                 elif kp['name'] == 'Mitad Columna':
#                     mitad_columna_pos = kp['position']
#                 elif kp['name'] == 'Base Cola':
#                     base_cola_pos = kp['position']

#             # Verificar si "Nariz" está presente
#             if nariz_pos:
#                 # Variable para almacenar los IDs de los objetos en cuya zona expandida está la "Nariz"
#                 objetos_con_nariz_en_zona = []

#                 # Verificar si la "Nariz" está dentro de alguna zona expandida de Botellas
#                 for obj_id, points_dilated in zona_botellas.items():
#                     polygon = Polygon(points_dilated)
#                     if not polygon.is_valid:
#                         polygon = polygon.buffer(0)
#                     if polygon.contains(Point(nariz_pos)):
#                         objetos_con_nariz_en_zona.append(obj_id)

#                 # Verificar si la "Nariz" está dentro de alguna zona expandida de Rastis
#                 for obj_id, points_dilated in zona_rastis.items():
#                     polygon = Polygon(points_dilated)
#                     if not polygon.is_valid:
#                         polygon = polygon.buffer(0)
#                     if polygon.contains(Point(nariz_pos)):
#                         objetos_con_nariz_en_zona.append(obj_id)

#                 # Si la "Nariz" está en alguna zona expandida, realizar las comprobaciones adicionales
#                 if objetos_con_nariz_en_zona:
#                     # Verificar si el ratón está sobre algún objeto (Regla 3)
#                     esta_sobre_objeto = False
#                     keypoints_para_verificar = [oreja_izq_pos, oreja_der_pos, nuca_pos, mitad_columna_pos, base_cola_pos]
#                     for class_name in ['Botella', 'Rasti']:
#                         if class_name in current_objects:
#                             contours = current_objects[class_name]
#                             for contour in contours:
#                                 contour_np = contour.reshape(-1, 2)
#                                 polygon = Polygon(contour_np)
#                                 if not polygon.is_valid:
#                                     polygon = polygon.buffer(0)
#                                 for kp_pos in keypoints_para_verificar:
#                                     if kp_pos and polygon.contains(Point(kp_pos)):
#                                         esta_sobre_objeto = True
#                                         break
#                                 if esta_sobre_objeto:
#                                     break
#                         if esta_sobre_objeto:
#                             break

#                     if not esta_sobre_objeto:
#                         # Verificar si "Nuca" está presente
#                         if nuca_pos:
#                             # Crear una línea desde "Nariz" y extenderla en la dirección de "Nariz" a "Nuca"
#                             line_length = 1000  # Longitud para extender la línea hacia adelante
#                             dx = nariz_pos[0] - nuca_pos[0]
#                             dy = nariz_pos[1] - nuca_pos[1]
#                             magnitude = np.sqrt(dx**2 + dy**2)
#                             if magnitude == 0:
#                                 direction = (0, 0)
#                             else:
#                                 direction = (dx / magnitude, dy / magnitude)
#                             # Extender la línea desde "Nariz" en la dirección hacia adelante
#                             extended_point = (nariz_pos[0] + direction[0] * line_length, nariz_pos[1] + direction[1] * line_length)
#                             line = LineString([nariz_pos, extended_point])

#                             # Ahora verificar si la línea interseca con el objeto correspondiente
#                             for obj_id in objetos_con_nariz_en_zona:
#                                 # Obtener el polígono del objeto
#                                 if obj_id in zona_botellas:
#                                     polygon = Polygon(zona_botellas[obj_id])
#                                 elif obj_id in zona_rastis:
#                                     polygon = Polygon(zona_rastis[obj_id])
#                                 else:
#                                     continue  # Si el objeto no está en las zonas, continuar

#                                 if not polygon.is_valid:
#                                     polygon = polygon.buffer(0)

#                                 if line.intersects(polygon):
#                                     # Contabilizar el tiempo
#                                     if obj_id not in tiempo_zonas:
#                                         tiempo_zonas[obj_id] = 0.0
#                                     tiempo_zonas[obj_id] += 1 / fps  # Incrementar tiempo en segundos

#                                     # Opcional: Dibujar la línea de dirección en el frame para visualización
#                                     cv2.line(final_frame, 
#                                         (int(nariz_pos[0]), int(nariz_pos[1])), 
#                                         (int(extended_point[0]), int(extended_point[1])), 
#                                         (0, 255, 0), 2)
#                                 else:
#                                     cv2.line(final_frame, 
#                                         (int(nariz_pos[0]), int(nariz_pos[1])), 
#                                         (int(extended_point[0]), int(extended_point[1])), 
#                                         (0, 0, 255), 2)

#                         else:
#                             print("Keypoint 'Nuca' no detectado en este frame.")
#                     else:
#                         print("El ratón está sobre un objeto. No se contabiliza el tiempo.")
            
#             # Mostrar el frame procesado
#             cv2.imshow('Detecciones Combinadas', final_frame)
            
#             # Guardar el frame en el video de salida si se especifica
#             if out:
#                 out.write(final_frame)
            
#             # Actualizar la barra de progreso
#             pbar.update(1)
            
#             # Salir con 'q'
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("Procesamiento interrumpido por el usuario.")
#                 break
    
#     # Liberar recursos después del procesamiento
#     cap.release()
#     if out:
#         out.release()
#     cv2.destroyAllWindows()
#     print("Procesamiento completado.")
    
#     # Post-procesamiento: Generar mapas de trayectoria y guardar datos
#     post_processing(keypoint_trajectories,
#                     distance_traveled,
#                     tiempo_zonas,
#                     trajectories_dir,
#                     video_name,
#                     current_objects, 
#                     zona_botellas, 
#                     zona_rastis,
#                     class_colors_matplotlib)

# if __name__ == "__main__":
#     # Rutas de ejemplo (ajusta según tu estructura)
#     video_path = "data/videos/Test_2_H_Recortado.mp4"  # Ruta al video de entrada
#     #video_path = "data/videos/Test_1_F_Recortado.mp4"  # Ruta al video de entrada
#     #video_path = "data/videos/Reconocimiento_2_H_Recortado.mp4"  # Ruta al video de entrada
#     keypoint_config = "models/keypoint_detection/config/config.yaml"  # Ruta al config de keypoints
#     keypoint_model_path = "models/keypoint_detection/Bests/best_model.pth.tar"  # Ruta al modelo de keypoints
#     yolo_model_path = "models/yolov11_segmentation/yolov11x_segmentation/weights/best.pt"  # Ruta al modelo YOLO
    
#     # Ejecutar la función principal con post-procesamiento
#     main(video_path, keypoint_config, keypoint_model_path, yolo_model_path)


########################################################################################

import sys
import os
import cv2
import torch
import yaml
import numpy as np
from tqdm import tqdm  # Importar tqdm para la barra de progreso
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon, Point
from collections import defaultdict

# Añadir el directorio padre al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Importar funciones de inferencia
from models.keypoint_detection.infer_keypoints import load_keypoint_model, get_keypoints, draw_keypoints
from models.yolov11_segmentation.infer_yolo import load_yolo_model, get_yolo_detections, draw_yolo_detections

def expand_mask_shapely(mask, escala, ampliacion_cm=10):
    """
    Expande la máscara añadiendo una distancia específica en centímetros a cada lado utilizando Shapely.

    Args:
        mask (torch.Tensor): Máscara original.
        escala (float): Escala en píxeles por centímetro (px/cm).
        ampliacion_cm (float): Distancia a añadir en centímetros.

    Returns:
        list: Lista de puntos del polígono expandido.
    """
    # Convertir la máscara a NumPy
    mask_numpy = mask.cpu().numpy()
    if mask_numpy.ndim == 3 and mask_numpy.shape[0] == 1:
        mask_numpy = mask_numpy.squeeze(0)  # De [1, H, W] a [H, W]
    mask_binary = (mask_numpy * 255).astype(np.uint8)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Advertencia: No se encontraron contornos en la máscara.")
        return []

    # Asumir que el contorno más grande es el objeto actual
    largest_contour = max(contours, key=cv2.contourArea)

    # Convertir el contorno a una lista de puntos
    points = [tuple(point[0]) for point in largest_contour]

    # Crear un polígono de Shapely
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # Corregir si es inválido

    # Ampliar el polígono
    buffer_distance = ampliacion_cm * escala  # Convertir cm a píxeles
    polygon_expanded = polygon.buffer(buffer_distance)

    if not polygon_expanded.is_valid:
        print("Advertencia: Polígono expandido inválido.")
        return []

    # Obtener los puntos del exterior del polígono expandido
    exterior_coords = np.array(polygon_expanded.exterior.coords).astype(int).tolist()

    return exterior_coords

# Definir las funciones de padding y redimensión
def pad_to_square(image, desired_size=1920, color=[0, 0, 0]):
    """
    Añade padding a la imagen para hacerla cuadrada, adaptando el tamaño si se especifica.

    Args:
        image (numpy.ndarray): Imagen original en formato BGR.
        desired_size (int): Tamaño deseado para el lado cuadrado.
        color (list): Color del padding en formato BGR.

    Returns:
        numpy.ndarray: Imagen cuadrada con padding.
    """
    old_size = image.shape[:2]  # (altura, ancho)
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

def resize_image(image, size=(640, 640)):
    """
    Redimensiona la imagen a un tamaño específico.

    Args:
        image (numpy.ndarray): Imagen en formato BGR.
        size (tuple): Nueva resolución (ancho, alto).

    Returns:
        numpy.ndarray: Imagen redimensionada.
    """
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized

def pad_and_resize(image, desired_size=1920, final_size=(640, 640), color=[0, 0, 0]):
    """
    Aplica padding para hacer la imagen cuadrada y luego la redimensiona a la resolución final.

    Args:
        image (numpy.ndarray): Imagen original.
        desired_size (int): Tamaño deseado para el lado cuadrado.
        final_size (tuple): Resolución final (ancho, alto).
        color (list): Color del padding en formato BGR.

    Returns:
        numpy.ndarray: Imagen redimensionada a la resolución final con padding aplicado.
    """
    padded_image = pad_to_square(image, desired_size, color)
    resized_image = resize_image(padded_image, final_size)
    return resized_image

def evaluate_keypoints(keypoints, image_shape=(640, 640)):
    """
    Evalúa la validez de los keypoints detectados.
    Por ejemplo, verifica si las coordenadas están dentro del rango de la imagen.

    Args:
        keypoints (list): Lista de keypoints detectados.
        image_shape (tuple): Tamaño de la imagen (alto, ancho).
    """
    for kp in keypoints:
        x, y = kp['position']
        if not (0 <= x < image_shape[1] and 0 <= y < image_shape[0]):
            print(f"Keypoint {kp['name']} fuera de los límites: ({x}, {y})")

def calculate_bounds(keypoint_trajectories, current_objects, zona_botellas, zona_rastis, padding=50):
    """
    Calcula los límites mínimos y máximos de las coordenadas para ajustar el zoom.

    Args:
        keypoint_trajectories (dict): Trayectorias de keypoints.
        current_objects (dict): Objetos detectados con sus contornos.
        zona_botellas (dict): Zonas ampliadas de las Botellas.
        zona_rastis (dict): Zonas ampliadas de los Rastis.
        padding (int): Espacio adicional alrededor de los puntos para el zoom.

    Returns:
        (xmin, xmax, ymin, ymax)
    """
    all_x = []
    all_y = []

    # Keypoints
    for positions in keypoint_trajectories.values():
        for pos in positions:
            all_x.append(pos[0])
            all_y.append(pos[1])

    # Objetos
    for obj_data in current_objects.values():
        contour = obj_data['approx']
        polygon = contour.reshape(-1, 2)
        all_x.extend(polygon[:, 0])
        all_y.extend(polygon[:, 1])

    # Zonas Botellas
    for points_dilated in zona_botellas.values():
        points = np.array(points_dilated)
        all_x.extend(points[:, 0])
        all_y.extend(points[:, 1])

    # Zonas Rastis
    for points_dilated in zona_rastis.values():
        points = np.array(points_dilated)
        all_x.extend(points[:, 0])
        all_y.extend(points[:, 1])

    xmin, xmax = min(all_x) - padding, max(all_x) + padding
    ymin, ymax = min(all_y) - padding, max(all_y) + padding

    # Asegurar que los límites no salgan de la imagen
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, 640)
    ymax = min(ymax, 640)

    return xmin, xmax, ymin, ymax

def post_processing(keypoint_trajectories,
                    distance_traveled,
                    tiempo_zonas,
                    trajectories_dir,
                    video_name,
                    current_objects, 
                    zona_botellas, 
                    zona_rastis,
                    class_colors):
    """
    Genera mapas de trayectoria que incluyen la base de la caja, los objetos y sus zonas ampliadas.

    Args:
        keypoint_trajectories (dict): Trayectorias de keypoints.
        distance_traveled (dict): Distancia recorrida por keypoints.
        tiempo_zonas (dict): Tiempo pasado en zonas de objetos.
        trajectories_dir (str): Directorio para guardar los resultados.
        video_name (str): Nombre del video procesado.
        current_objects (dict): Objetos detectados con sus contornos.
        zona_botellas (dict): Zonas ampliadas de las Botellas.
        zona_rastis (dict): Zonas ampliadas de los Rastis.
        class_colors (dict): Colores para las clases en matplotlib.
    """
    # Calcular los límites para el zoom
    xmin, xmax, ymin, ymax = calculate_bounds(keypoint_trajectories, current_objects, zona_botellas, zona_rastis)

    # Dibujar las trayectorias de keypoints
    for kp_name, positions in keypoint_trajectories.items():
        # Crear una figura con fondo blanco
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('white')  # Fondo blanco
        # Mantener la proporción de aspecto
        ax.set_aspect('equal')

        # Establecer límites del gráfico con zoom
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)  # Invertir el eje Y para que coincida con la imagen

        # Ocultar los ejes
        ax.axis('off')

        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        ax.plot(x, y, marker='o', markersize=2, label=f'Trayectoria {kp_name}')

        # Añadir etiqueta de distancia
        distancia = distance_traveled.get(kp_name, 0.0)
        ax.text(xmin + 10, ymin + 30, 
                f"Distancia {kp_name}: {distancia:.2f} m", 
                color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # Dibujar los objetos detectados
        class_labels = {'BaseCaja': 'BaseCaja', 'Botella': 'Botella', 'Rasti': 'Rasti'}
        class_colors_plot = class_colors

        # Usar un diccionario para mantener las leyendas únicas
        legend_handles = {}

        for obj_id, obj_data in current_objects.items():
            class_name = obj_data['class_name']
            contour = obj_data['approx'].reshape(-1, 2)
            
            # Asegurar que el polígono esté cerrado
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0]])
            
            # Crear un polígono y añadirlo al plot
            polygon = MplPolygon(contour, closed=True, fill=None, edgecolor=class_colors_plot[class_name], linewidth=2, label=class_labels[class_name])
            ax.add_patch(polygon)

            # Añadir al diccionario de handles para leyenda (solo una vez por clase)
            if class_name not in legend_handles:
                legend_handles[class_name] = patches.Patch(edgecolor=class_colors_plot[class_name], facecolor='none', label=class_labels[class_name])
            
            if class_name != "BaseCaja":
                # **Calcular el centroide del objeto y añadir una etiqueta con obj_id**
                centroid = np.mean(contour, axis=0)
                ax.text(centroid[0], centroid[1], obj_id, fontsize=6, fontweight='bold', ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Dibujar zonas extendidas (Botellas y Rastis)
        for zona_id, points_dilated in zona_botellas.items():
            points = np.array(points_dilated)
            
            # Asegurar que el polígono esté cerrado
            if not np.array_equal(points[0], points[-1]):
                points = np.vstack([points, points[0]])
            
            polygon = MplPolygon(points, closed=True, fill=None, edgecolor='orange', linestyle='--', linewidth=1, label='Zona Botella')
            ax.add_patch(polygon)
            # Añadir al diccionario de handles para leyenda
            if 'Zona Botella' not in legend_handles:
                legend_handles['Zona Botella'] = patches.Patch(edgecolor='orange', facecolor='none', linestyle='--', label='Zona Botella')

        for zona_id, points_dilated in zona_rastis.items():
            points = np.array(points_dilated)
            
            # Asegurar que el polígono esté cerrado
            if not np.array_equal(points[0], points[-1]):
                points = np.vstack([points, points[0]])
            
            polygon = MplPolygon(points, closed=True, fill=None, edgecolor='blue', linestyle='--', linewidth=1, label='Zona Rasti')
            ax.add_patch(polygon)
            # Añadir al diccionario de handles para leyenda
            if 'Zona Rasti' not in legend_handles:
                legend_handles['Zona Rasti'] = patches.Patch(edgecolor='blue', facecolor='none', linestyle='--', label='Zona Rasti')

        # Añadir leyenda fuera del gráfico
        ax.legend(handles=legend_handles.values(), loc='upper right', bbox_to_anchor=(1.10, 1.05), borderaxespad=0.)

        # Título y etiquetas
        ax.set_title(f'Mapa de Trayectoria {kp_name} - {video_name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Guardar la figura
        plt.savefig(os.path.join(trajectories_dir, f'Trayectoria_{kp_name}_con_distancia.png'))
        plt.close()

    # Guardar distancias recorridas
    with open(os.path.join(trajectories_dir, 'distancias_recorridas.txt'), 'w') as f:
        for kp_name, distancia in distance_traveled.items():
            f.write(f"Keypoint {kp_name}: {distancia:.4f} metros\n")

    # Guardar tiempos en zonas
    with open(os.path.join(trajectories_dir, 'tiempos_zonas.txt'), 'w') as f:
        total_time = 0.0
        for obj_id, tiempo in tiempo_zonas.items():
            f.write(f"Objeto {obj_id}: {tiempo:.4f} segundos\n")
            total_time += tiempo
        f.write(f"Tiempo total en zonas: {total_time:.4f} segundos\n")

def main(video_path, keypoint_config, keypoint_model_path, yolo_model_path):
    # Crear directorio de salida para mapas de trayectoria
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    trajectories_dir = os.path.join("outputs", f"processed_{video_name}")
    os.makedirs(trajectories_dir, exist_ok=True)

    output_path = f"outputs/processed_{video_name}/{video_name}_processed.mp4"  # Ruta al video de salida (opcional)
    # Asegúrate de que el directorio de salida existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Variables para almacenar datos
    keypoint_trajectories = {}  # Diccionario para almacenar trayectorias de keypoints
    distance_traveled = {}       # Diccionario para almacenar distancia recorrida
    tiempo_zonas = {}            # Diccionario para almacenar tiempo en zonas de objetos
    current_objects = {}         # Diccionario para almacenar objetos detectados con sus contornos
    zona_botellas = {}           # Diccionario para almacenar zonas de Botellas
    zona_rastis = {}             # Diccionario para almacenar zonas de Rastis
    escala = 0.0                 # Escala: px/cm, se calculará después de la detección inicial

    # Cargar modelos
    print("Cargando modelo de detección de keypoints...")
    keypoint_model, keypoint_device = load_keypoint_model(keypoint_config, keypoint_model_path)
    print("Modelo de keypoints cargado.")

    print("Cargando modelo de segmentación YOLOv11...")
    yolo_model = load_yolo_model(yolo_model_path)
    print("Modelo YOLOv11 cargado.")

    # Definir colores para YOLO en formato BGR
    class_colors = {
        'BaseCaja': (128, 128, 128),  # Gris
        'Botella': (0, 165, 255),     # Naranja
        'Rasti': (255, 0, 0),         # Azul
        # Añade más clases si es necesario
    }

    class_colors_matplotlib = {
        'BaseCaja': 'gray',
        'Botella': 'orange',
        'Rasti': 'blue',
    }

    target_classes = ['BaseCaja', 'Botella', 'Rasti']
    target_class_name = 'BaseCaja'

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    # Obtener propiedades del video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolución del video: {frame_width}x{frame_height} píxeles")
    print(f"FPS del video: {fps}")
    print(f"Número total de frames: {total_frames}")

    # Definir el codec y crear VideoWriter si se desea guardar el output
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Cambia el codec si es necesario
        out = cv2.VideoWriter(output_path, fourcc, fps, (640, 640))
        print(f"Video de salida guardado en: {output_path}")
    else:
        out = None

    # Iniciar la barra de progreso
    with tqdm(total=total_frames, desc="Procesando Video", unit="frame") as pbar:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"\nProcesando frame {frame_count}/{total_frames}")

            # Aplicar padding y redimensión
            processed_frame = pad_and_resize(frame, desired_size=1920, final_size=(640, 640), color=[0, 0, 0])

            if frame_count == 1:
                # Realizar detección con YOLO solo en el primer frame
                yolo_results = get_yolo_detections(yolo_model, processed_frame)
                print(f"Resultado: {yolo_results}")

                # Extraer detecciones de YOLO para procesar 'BaseCaja', 'Botella' y 'Rasti'
                detections = []
                for result in yolo_results:
                    masks = result.masks
                    boxes = result.boxes
                    if masks is None or boxes is None:
                        continue
                    for mask, box in zip(masks.data, boxes):
                        class_id = int(box.cls.cpu().numpy())
                        class_name = yolo_model.names[class_id]
                        bbox = box.xyxy.cpu().numpy().astype(int).tolist()  # [x1, y1, x2, y2]
                        detections.append({'class': class_name, 'bbox': bbox, 'mask': mask})

                # Identificar BaseCaja para calcular la escala
                base_caja = next((obj for obj in detections if obj['class'] == 'BaseCaja'), None)
                if base_caja:
                    mask = base_caja['mask']  # La máscara correspondiente a BaseCaja
                    try:
                        # Convertir la máscara a un arreglo de NumPy
                        mask_numpy = mask.cpu().numpy()
                        if mask_numpy.ndim == 3 and mask_numpy.shape[0] == 1:
                            mask_numpy = mask_numpy.squeeze(0)  # Convertir de [1, H, W] a [H, W]
                        mask_binary = (mask_numpy * 255).astype(np.uint8)

                        # Guardar la máscara para depuración (opcional)
                        cv2.imwrite(os.path.join(trajectories_dir, f"mask_base_caja_frame{frame_count}.png"), mask_binary)

                        # Encontrar contornos en la máscara
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            print("Error: No se encontraron contornos en la máscara de BaseCaja.")
                            return

                        # Asumir que el contorno más grande es BaseCaja
                        largest_contour = max(contours, key=cv2.contourArea)

                        # Aproximar el contorno a un polígono con menos puntos
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                        # Verificar que tenemos cuatro puntos
                        if len(approx) != 4:
                            print(f"Advertencia: Se encontraron {len(approx)} puntos en el contorno de BaseCaja, se esperaban 4.")

                        # Extraer los puntos
                        points = [point[0] for point in approx]
                        print(f"Puntos del contorno de BaseCaja: {points}")  # Depuración

                        approx = approx.reshape(-1, 1, 2).astype(np.int32)

                        # Almacenar BaseCaja en current_objects
                        current_objects['BaseCaja'] = {'class_name': 'BaseCaja', 'approx': approx}

                        # Calcular las distancias entre puntos consecutivos
                        distances = []
                        for i in range(len(points)):
                            p1 = points[i]
                            p2 = points[(i + 1) % len(points)]
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            distance = np.sqrt(dx**2 + dy**2)
                            distances.append(distance)
                            print(f"Distancia entre {p1} y {p2}: {distance} px")  # Depuración

                        # Calcular la escala como el promedio de las distancias dividido por el tamaño real
                        average_distance_px = np.mean(distances)
                        real_size_cm = 40  # Tamaño real de BaseCaja en cm
                        escala = average_distance_px / real_size_cm  # px/cm
                        print(f"Escala calculada: {escala:.2f} px/cm")
                    except AttributeError:
                        print("Error: No se pudo acceder al tensor de la máscara de BaseCaja.")
                        return
                else:
                    print("Error: No se detectó BaseCaja en el primer frame para calcular la escala.")
                    return

                # Procesar las demás clases: Botella y Rasti
                obj_counter = {'Botella': 0, 'Rasti': 0}
                for obj in detections:
                    class_name = obj['class']
                    if class_name not in ['Botella', 'Rasti']:
                        continue  # Solo procesar Botella y Rasti

                    mask = obj['mask']
                    try:
                        # Incrementar el contador de objetos
                        obj_counter[class_name] += 1
                        obj_id = f"{class_name}_{obj_counter[class_name]}"

                        # Expandir la máscara y obtener el polígono dilatado
                        points_dilated = expand_mask_shapely(mask, escala, ampliacion_cm=3)
                        if not points_dilated:
                            continue  # Saltar si no se pudo expandir

                        print(f"Puntos del contorno dilatado de {class_name}: {points_dilated}")  # Depuración

                        # Dibujar el polígono expandido en el frame
                        cv2.polylines(processed_frame, [np.array(points_dilated)], isClosed=True, color=class_colors[class_name], thickness=2)

                        # Almacenar las zonas ampliadas con el obj_id
                        if class_name == 'Botella':
                            zona_botellas[obj_id] = points_dilated
                        elif class_name == 'Rasti':
                            zona_rastis[obj_id] = points_dilated

                        mask_numpy = mask.cpu().numpy()
                        if mask_numpy.ndim == 3 and mask_numpy.shape[0] == 1:
                            mask_numpy = mask_numpy.squeeze(0)  # Convertir de [1, H, W] a [H, W]
                        mask_binary = (mask_numpy * 255).astype(np.uint8)

                        # Encontrar contornos en la máscara
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            print(f"Error: No se encontraron contornos en la máscara de {class_name}.")
                            continue

                        largest_contour = max(contours, key=cv2.contourArea)

                        # Aproximar el contorno a un polígono con menos puntos
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                        approx = approx.reshape(-1, 1, 2).astype(np.int32)
                        # Almacenar el approx y class_name en current_objects con el obj_id
                        current_objects[obj_id] = {'class_name': class_name, 'approx': approx}

                    except AttributeError:
                        print(f"Error: No se pudo acceder al tensor de la máscara de {class_name}.")
                        continue

                final_frame = processed_frame.copy()

                # Procesar con el modelo de keypoints
                keypoints = get_keypoints(keypoint_model, keypoint_device, processed_frame)
                evaluate_keypoints(keypoints)
                # Dibujar keypoints
                draw_keypoints(final_frame, keypoints)

                # Dibujar objetos detectados
                for obj_id, obj_data in current_objects.items():
                    class_name = obj_data['class_name']
                    approx = [obj_data['approx']]
                    cv2.polylines(final_frame, approx, isClosed=True, color=class_colors[class_name], thickness=2)

            else:
                # Procesar con el modelo de keypoints
                keypoints = get_keypoints(keypoint_model, keypoint_device, processed_frame)
                evaluate_keypoints(keypoints)
                #print(f"Keypoints detectados: {keypoints}")  # Depuración

                # Para frames posteriores, dibujar las zonas ampliadas almacenadas
                final_frame = processed_frame.copy()
                # Dibujar keypoints
                draw_keypoints(final_frame, keypoints)

                # Dibujar objetos detectados
                for obj_id, obj_data in current_objects.items():
                    class_name = obj_data['class_name']
                    approx = [obj_data['approx']]
                    cv2.polylines(final_frame, approx, isClosed=True, color=class_colors[class_name], thickness=2)

                # Dibujar zonas expandidas directamente
                for obj_id, points_dilated in zona_botellas.items():
                    cv2.polylines(final_frame, [np.array(points_dilated, dtype=np.int32)], isClosed=True, color=class_colors['Botella'], thickness=2)
                for obj_id, points_dilated in zona_rastis.items():
                    cv2.polylines(final_frame, [np.array(points_dilated, dtype=np.int32)], isClosed=True, color=class_colors['Rasti'], thickness=2)

            # Actualizar trayectorias de keypoints
            for kp in keypoints:
                name = kp['name']
                pos = kp['position']
                if name not in keypoint_trajectories:
                    keypoint_trajectories[name] = []
                keypoint_trajectories[name].append(pos)

                # Calcular distancia recorrida
                if name not in distance_traveled:
                    distance_traveled[name] = 0.0
                if len(keypoint_trajectories[name]) > 1:
                    prev_pos = keypoint_trajectories[name][-2]
                    dx = pos[0] - prev_pos[0]
                    dy = pos[1] - prev_pos[1]
                    distance = np.sqrt(dx**2 + dy**2) / escala / 100  # metros
                    distance_traveled[name] += distance

            # Inicializar variables de posiciones de keypoints
            nariz_pos = None
            nuca_pos = None
            oreja_izq_pos = None
            oreja_der_pos = None
            mitad_columna_pos = None
            base_cola_pos = None

            # Obtener posiciones de keypoints
            for kp in keypoints:
                if kp['name'] == 'Nariz':
                    nariz_pos = kp['position']
                elif kp['name'] == 'Nuca':
                    nuca_pos = kp['position']
                elif kp['name'] == 'Oreja Izquierda':
                    oreja_izq_pos = kp['position']
                elif kp['name'] == 'Oreja Derecha':
                    oreja_der_pos = kp['position']
                elif kp['name'] == 'Mitad Columna':
                    mitad_columna_pos = kp['position']
                elif kp['name'] == 'Base Cola':
                    base_cola_pos = kp['position']

            # Inicializar line_color en rojo (no se contabiliza el tiempo)
            line_color = (0, 0, 255)  # Rojo en BGR

            # Verificar si "Nariz" está presente
            if nariz_pos:
                # Variable para almacenar los IDs de los objetos en cuya zona expandida está la "Nariz"
                objetos_con_nariz_en_zona = []

                # Verificar si la "Nariz" está dentro de alguna zona ampliada de Botellas
                for obj_id, points_dilated in zona_botellas.items():
                    polygon = Polygon(points_dilated)
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    if polygon.contains(Point(nariz_pos)):
                        objetos_con_nariz_en_zona.append(obj_id)

                # Verificar si la "Nariz" está dentro de alguna zona ampliada de Rastis
                for obj_id, points_dilated in zona_rastis.items():
                    polygon = Polygon(points_dilated)
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    if polygon.contains(Point(nariz_pos)):
                        objetos_con_nariz_en_zona.append(obj_id)

                # Si la "Nariz" está en alguna zona ampliada, realizar las comprobaciones adicionales
                if objetos_con_nariz_en_zona:
                    # Verificar si el ratón está sobre algún objeto (Regla 3)
                    esta_sobre_objeto = False
                    keypoints_para_verificar = [oreja_izq_pos, oreja_der_pos, nuca_pos, mitad_columna_pos, base_cola_pos]
                    for class_name in ['Botella', 'Rasti']:
                        for obj_id, obj_data in current_objects.items():
                            if obj_data['class_name'] == class_name:
                                contour_np = obj_data['approx'].reshape(-1, 2)
                                polygon = Polygon(contour_np)
                                if not polygon.is_valid:
                                    polygon = polygon.buffer(0)
                                for kp_pos in keypoints_para_verificar:
                                    if kp_pos and polygon.contains(Point(kp_pos)):
                                        esta_sobre_objeto = True
                                        break
                                if esta_sobre_objeto:
                                    break
                        if esta_sobre_objeto:
                            break

                    if not esta_sobre_objeto:
                        # Verificar si "Nuca" está presente
                        if nuca_pos:
                            # Crear una línea desde "Nariz" y extenderla en la dirección de "Nuca" a "Nariz"
                            line_length = 1000  # Longitud para extender la línea hacia adelante
                            dx = nariz_pos[0] - nuca_pos[0]
                            dy = nariz_pos[1] - nuca_pos[1]
                            magnitude = np.sqrt(dx**2 + dy**2)
                            if magnitude == 0:
                                direction = (0, 0)
                            else:
                                direction = (dx / magnitude, dy / magnitude)
                            # Extender la línea desde "Nariz" en la dirección hacia adelante
                            extended_point = (nariz_pos[0] + direction[0] * line_length, nariz_pos[1] + direction[1] * line_length)

                            # Inicializar variable para saber si se contabiliza el tiempo
                            tiempo_contabilizado = False

                            # Ahora verificar si la línea interseca con el objeto correspondiente
                            for obj_id in objetos_con_nariz_en_zona:
                                # Obtener el contorno real del objeto desde current_objects
                                if obj_id in current_objects:
                                    obj_data = current_objects[obj_id]
                                    approx = obj_data['approx']
                                    class_name = obj_data['class_name']
                                    # Convertir approx a un array de numpy de forma (N, 2)
                                    contour_np = approx.reshape(-1, 2)
                                    polygon = Polygon(contour_np)
                                else:
                                    continue  # Si el objeto no está en current_objects, continuar

                                if not polygon.is_valid:
                                    polygon = polygon.buffer(0)

                                line = LineString([nariz_pos, extended_point])

                                if line.intersects(polygon):
                                    # Contabilizar el tiempo
                                    if obj_id not in tiempo_zonas:
                                        tiempo_zonas[obj_id] = 0.0
                                    tiempo_zonas[obj_id] += 1 / fps  # Incrementar tiempo en segundos

                                    # Cambiar el color de la línea a verde (se contabiliza el tiempo)
                                    line_color = (0, 255, 0)  # Verde en BGR
                                    tiempo_contabilizado = True
                                    break  # Ya que hemos contabilizado el tiempo, podemos salir del bucle

                            # Dibujar la línea de dirección en el frame final con el color correspondiente
                            cv2.line(final_frame,
                                     (int(nariz_pos[0]), int(nariz_pos[1])),
                                     (int(extended_point[0]), int(extended_point[1])),
                                     line_color, 2)
                        else:
                            print("Keypoint 'Nuca' no detectado en este frame.")
                    else:
                        print("El ratón está sobre un objeto. No se contabiliza el tiempo.")
                        # Dibujar la línea en rojo ya que no se está contabilizando el tiempo
                        if nuca_pos:
                            # Crear una línea desde "Nariz" y extenderla en la dirección de "Nuca" a "Nariz"
                            line_length = 1000
                            dx = nariz_pos[0] - nuca_pos[0]
                            dy = nariz_pos[1] - nuca_pos[1]
                            magnitude = np.sqrt(dx**2 + dy**2)
                            if magnitude == 0:
                                direction = (0, 0)
                            else:
                                direction = (dx / magnitude, dy / magnitude)
                            extended_point = (nariz_pos[0] + direction[0] * line_length, nariz_pos[1] + direction[1] * line_length)

                            cv2.line(final_frame,
                                     (int(nariz_pos[0]), int(nariz_pos[1])),
                                     (int(extended_point[0]), int(extended_point[1])),
                                     line_color, 2)
                else:
                    # Si la "Nariz" no está en ninguna zona ampliada
                    if nuca_pos:
                        # Crear una línea desde "Nariz" y extenderla en la dirección de "Nuca" a "Nariz"
                        line_length = 1000
                        dx = nariz_pos[0] - nuca_pos[0]
                        dy = nariz_pos[1] - nuca_pos[1]
                        magnitude = np.sqrt(dx**2 + dy**2)
                        if magnitude == 0:
                            direction = (0, 0)
                        else:
                            direction = (dx / magnitude, dy / magnitude)
                        extended_point = (nariz_pos[0] + direction[0] * line_length, nariz_pos[1] + direction[1] * line_length)

                        # Dibujar la línea en rojo
                        cv2.line(final_frame,
                                 (int(nariz_pos[0]), int(nariz_pos[1])),
                                 (int(extended_point[0]), int(extended_point[1])),
                                 line_color, 2)

            # Mostrar el frame procesado
            cv2.imshow('Detecciones Combinadas', final_frame)

            # Guardar el frame en el video de salida si se especifica
            if out:
                out.write(final_frame)

            # Actualizar la barra de progreso
            pbar.update(1)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Procesamiento interrumpido por el usuario.")
                break

    # Liberar recursos después del procesamiento
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("Procesamiento completado.")

    # Post-procesamiento: Generar mapas de trayectoria y guardar datos
    post_processing(keypoint_trajectories,
                    distance_traveled,
                    tiempo_zonas,
                    trajectories_dir,
                    video_name,
                    current_objects, 
                    zona_botellas, 
                    zona_rastis,
                    class_colors_matplotlib)

if __name__ == "__main__":
    # Rutas de ejemplo (ajusta según tu estructura)
    #video_path = "data/videos/Test_2_H_Recortado.mp4"  # Ruta al video de entrada
    video_path = "data/videos/Test_1_F_Recortado.mp4"  # Ruta al video de entrada
    #video_path = "data/videos/Reconocimiento_2_H_Recortado.mp4"  # Ruta al video de entrada
    keypoint_config = "models/keypoint_detection/config/config.yaml"  # Ruta al config de keypoints
    keypoint_model_path = "models/keypoint_detection/Bests/best_model.pth.tar"  # Ruta al modelo de keypoints
    yolo_model_path = "models/yolov11_segmentation/yolov11x_segmentation/weights/best.pt"  # Ruta al modelo YOLO

    # Ejecutar la función principal con post-procesamiento
    main(video_path, keypoint_config, keypoint_model_path, yolo_model_path)
