import sys
import os
import cv2
import torch
import yaml
import argparse
import time
import numpy as np
from tqdm import tqdm  # Importar tqdm para la barra de progreso
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon, Point
from collections import defaultdict
from save_db import insert_results_to_db
from io import BytesIO  # Importar para trabajar con binarios en memoria
import tempfile

# Añadir el directorio padre al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Importar funciones de inferencia
from models.keypoint_detection.infer_keypoints import load_keypoint_model, get_keypoints, draw_keypoints
from models.yolov11_segmentation.infer_yolo import load_yolo_model, get_yolo_detections, draw_yolo_detections

analysis_should_continue = True

def closeCv2Window(cap, out):
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

def should_continue_analysis():
    global analysis_should_continue
    return analysis_should_continue

def cancel_analysis():
    global analysis_should_continue
    analysis_should_continue = False

def shrink_polygon_towards_centroid(points, delta):
    """
    Reduce un polígono hacia su centroide moviendo cada punto una distancia delta en píxeles.

    Args:
        points (list): Lista de puntos (x, y) del polígono.
        delta (float): Distancia en píxeles para mover cada punto hacia el centroide.

    Returns:
        list: Nueva lista de puntos (x, y) del polígono reducido.
    """
    polygon = Polygon(points)
    centroid = polygon.centroid

    new_points = []
    for x, y in points:
        vec_x = centroid.x - x
        vec_y = centroid.y - y
        length = np.hypot(vec_x, vec_y)
        if length == 0:
            new_x, new_y = x, y  # El punto coincide con el centroide
        else:
            move_x = vec_x / length * delta
            move_y = vec_y / length * delta
            new_x = x + move_x
            new_y = y + move_y
        new_points.append((new_x, new_y))
    return new_points

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
                    keypoint_area_data,
                    trajectories_dir,
                    video_name,
                    current_objects, 
                    zona_botellas, 
                    zona_rastis,
                    zona_central,
                    class_colors,
                    id_raton,
                    id_dosis,
                    cantidad,
                    mail_usuario
                    ):
    """
    Genera mapas de trayectoria que incluyen la base de la caja, los objetos y sus zonas ampliadas.

    Args:
        keypoint_trajectories (dict): Trayectorias de keypoints.
        distance_traveled (dict): Distancia recorrida por keypoints.
        tiempo_zonas (dict): Tiempo pasado en zonas de objetos.
        keypoint_area_data (dict): Datos de los keypoints dentro del área central.
        trajectories_dir (str): Directorio para guardar los resultados.
        video_name (str): Nombre del video procesado.
        current_objects (dict): Objetos detectados con sus contornos.
        zona_botellas (dict): Zonas ampliadas de las Botellas.
        zona_rastis (dict): Zonas ampliadas de los Rastis.
        zona_central (list): Puntos del área central de la BaseCaja.
        class_colors (dict): Colores para las clases en matplotlib.
    """
    # Calcular los límites para el zoom
    xmin, xmax, ymin, ymax = calculate_bounds(keypoint_trajectories, current_objects, zona_botellas, zona_rastis)
    # Ajustar límites para incluir el área central
    if zona_central is not None:
        points = np.array(zona_central)
        all_x = points[:, 0]
        all_y = points[:, 1]
        xmin = min(xmin, min(all_x) - 50)
        xmax = max(xmax, max(all_x) + 50)
        ymin = min(ymin, min(all_y) - 50)
        ymax = max(ymax, max(all_y) + 50)
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, 640)
        ymax = min(ymax, 640)

    trajectory_data = []
    
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
        ax.text(xmin, ymin + 15, 
                f"Distancia {kp_name}: {distancia:.2f} m", 
                color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # Obtener la distancia recorrida dentro del área central
        distancia_central = keypoint_area_data.get(kp_name, {}).get('distance_inside', 0.0)
        ax.text(xmin , ymin + 35, 
                f"Distancia A.C. {kp_name}: {distancia_central:.2f} m", 
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

        # Dibujar el área central
        if zona_central is not None:
            points = np.array(zona_central)
            # Asegurar que el polígono esté cerrado
            if not np.array_equal(points[0], points[-1]):
                points = np.vstack([points, points[0]])
            polygon = MplPolygon(points, closed=True, fill=None, edgecolor='darkgray', linestyle='--', linewidth=2, label='Área Central')
            ax.add_patch(polygon)
            # Añadir al diccionario de handles para leyenda
            if 'Área Central' not in legend_handles:
                legend_handles['Área Central'] = patches.Patch(edgecolor='darkgray', facecolor='none', linestyle='--', label='Área Central')

        # Añadir leyenda fuera del gráfico
        ax.legend(handles=legend_handles.values(), loc='upper right', bbox_to_anchor=(1.10, 1.05), borderaxespad=0.)

        # Título y etiquetas
        ax.set_title(f'Mapa de Trayectoria {kp_name} - {video_name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Convertir el gráfico a binario
        buffer = BytesIO()
        plt.savefig(buffer, format='png')  # Guardar en el buffer
        plt.close(fig)  # Cerrar la figura
        buffer.seek(0)  # Volver al inicio del buffer
        trajectory_map = buffer.read()  # Leer los datos binarios

        # Agregar la información al diccionario que se enviará a la base de datos
        trajectory_data.append({
            'keypoint': kp_name,
            'distance': distance_traveled.get(kp_name, 0.0),
            'map': trajectory_map,
            'area_central': keypoint_area_data.get(kp_name, {}).get('distance_inside', 0.0),
            'entries': keypoint_area_data.get(kp_name, {}).get('enter_count', 0),
            'exits': keypoint_area_data.get(kp_name, {}).get('exit_count', 0),
        })

    print("Contenido de times_data:", tiempo_zonas)

    if should_continue_analysis():
        insert_results_to_db(
        distances=distance_traveled,
        area_central_data=keypoint_area_data,
        times_data=tiempo_zonas,
        trajectory_data=trajectory_data,  # Diccionario con los mapas generados en bytes
        video_name=video_name,
        id_raton=id_raton,
        id_dosis=id_dosis,
        cantidad=cantidad,
        mail_usuario=mail_usuario
        )



def main(video_path, keypoint_config, keypoint_model_path, yolo_model_path, id_raton, id_dosis, cantidad, mail_usuario, progress_callback=None):
    # Crear un directorio temporal para los mapas de trayectoria
    # Cuando el bloque `with` termina, el directorio temporal y su contenido se eliminan automáticamente.
    global analysis_should_continue
    analysis_should_continue = True

    if not should_continue_analysis():
        print("Análisis cancelado.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        trajectories_dir = os.path.join(temp_dir, f"processed_{video_name}")
        os.makedirs(trajectories_dir, exist_ok=True)

        print(f"Los mapas de trayectoria se almacenarán temporalmente en: {trajectories_dir}")

        output_path = f"outputs/processed_{video_name}/{video_name}_processed.mp4"  # Ruta al video de salida (opcional)
        # Asegúrate de que el directorio de salida existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Variables para almacenar datos
        keypoint_trajectories = {}  # Diccionario para almacenar trayectorias de keypoints
        distance_traveled = {}       # Diccionario para almacenar distancia recorrida
        tiempo_zonas = {}            # Diccionario para almacenar tiempo en zonas de objetos
        keypoint_area_data = {}      # Diccionario para datos del área central
        current_objects = {}         # Diccionario para almacenar objetos detectados con sus contornos
        zona_botellas = {}           # Diccionario para almacenar zonas de Botellas
        zona_rastis = {}             # Diccionario para almacenar zonas de Rastis
        escala = 0.0                 # Escala: px/cm, se calculará después de la detección inicial
        zona_central = None          # Zona central de la BaseCaja

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
        if not should_continue_analysis():
            print("Análisis cancelado.")
            closeCv2Window(cap, out)
            return
        start_time = time.time()  # Marca el inicio del procesamiento
        # Iniciar la barra de progreso
        with tqdm(total=total_frames, desc="Procesando Video", unit="frame") as pbar:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if not should_continue_analysis():
                    print("Análisis cancelado.")
                    closeCv2Window(cap, out)
                    return
                
                frame_count += 1
                print(f"\nProcesando frame {frame_count}/{total_frames}")

                # Aplicar padding y redimensión
                processed_frame = pad_and_resize(frame, desired_size=1920, final_size=(640, 640), color=[0, 0, 0])

                if frame_count == 1:
                    # Realizar detección con YOLO solo en el primer frame
                    yolo_results = get_yolo_detections(yolo_model, processed_frame)
                    print(f"Resultado: {yolo_results}")

                    # Extraer detecciones de YOLO para procesar 'BaseCaja', 'Botella' y 'Rasti'
                    detecciones = []
                    for result in yolo_results:
                        masks = result.masks
                        boxes = result.boxes
                        if masks is None or boxes is None:
                            continue
                        for mask, box in zip(masks.data, boxes):
                            class_id = int(box.cls.cpu().numpy())
                            class_name = yolo_model.names[class_id]
                            bbox = box.xyxy.cpu().numpy().astype(int).tolist()  # [x1, y1, x2, y2]
                            detecciones.append({'class': class_name, 'bbox': bbox, 'mask': mask})

                    # Identificar BaseCaja para calcular la escala
                    base_caja = next((obj for obj in detecciones if obj['class'] == 'BaseCaja'), None)
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

                            # Calcular el área central
                            delta_px = 10 * escala  # 20 cm hacia el centro
                            zona_central_points = shrink_polygon_towards_centroid(points, delta_px)
                            zona_central = np.array(zona_central_points, dtype=np.int32)

                        except AttributeError:
                            print("Error: No se pudo acceder al tensor de la máscara de BaseCaja.")
                            return
                    else:
                        print("Error: No se detectó BaseCaja en el primer frame para calcular la escala.")
                        return

                    # Procesar las demás clases: Botella y Rasti
                    obj_counter = {'Botella': 0, 'Rasti': 0}
                    for obj in detecciones:
                        class_name = obj['class']
                        if class_name not in ['Botella', 'Rasti']:
                            continue  # Solo procesar Botella y Rasti

                        mask = obj['mask']
                        try:
                            # Incrementar el contador de objetos
                            obj_counter[class_name] += 1
                            obj_id = f"{class_name}_{obj_counter[class_name]}"

                            # Expandir la máscara y obtener el polígono dilatado
                            points_dilated = expand_mask_shapely(mask, escala, ampliacion_cm=2)
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

                    # Dibujar el área central en final_frame
                    if zona_central is not None:
                        cv2.polylines(final_frame, [zona_central], isClosed=True, color=(169, 169, 169), thickness=2)

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
                    
                    # Dibujar el área central en final_frame
                    if zona_central is not None:
                        cv2.polylines(final_frame, [zona_central], isClosed=True, color=(169, 169, 169), thickness=2)

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

                # Verificar si los keypoints están dentro del área central
                if zona_central is not None:
                    zona_central_polygon = Polygon(zona_central_points)
                    if not zona_central_polygon.is_valid:
                        zona_central_polygon = zona_central_polygon.buffer(0)
                    for kp in keypoints:
                        name = kp['name']
                        pos = kp['position']
                        point = Point(pos)
                        is_inside = zona_central_polygon.contains(point)
                        if name not in keypoint_area_data:
                            keypoint_area_data[name] = {
                                'inside': False,
                                'enter_count': 0,   # Número de veces que entra
                                'exit_count': 0,    # Número de veces que sale
                                'time_inside': 0.0,
                                'distance_inside': 0.0,
                                'prev_pos_inside': None,
                            }
                        area_data = keypoint_area_data[name]
                        if is_inside:
                            if not area_data['inside']:
                                # El keypoint acaba de entrar al área
                                area_data['enter_count'] += 1
                            # Actualizar tiempo dentro del área
                            area_data['time_inside'] += 1 / fps
                            if area_data['prev_pos_inside'] is not None:
                                dx = pos[0] - area_data['prev_pos_inside'][0]
                                dy = pos[1] - area_data['prev_pos_inside'][1]
                                distance = np.hypot(dx, dy) / escala / 100  # Convertir a metros
                                area_data['distance_inside'] += distance
                            area_data['prev_pos_inside'] = pos
                            area_data['inside'] = True
                        else:
                            if area_data['inside']:
                                # El keypoint acaba de salir del área
                                area_data['exit_count'] += 1
                            area_data['inside'] = False
                            area_data['prev_pos_inside'] = None

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
                            # cv2.line(final_frame,
                            #          (int(nariz_pos[0]), int(nariz_pos[1])),
                            #          (int(extended_point[0]), int(extended_point[1])),
                            #          line_color, 2)

                # Mostrar el frame procesado
                cv2.imshow('Detecciones Combinadas', final_frame)

                # Guardar el frame en el video de salida si se especifica
                if out:
                    out.write(final_frame)

                # Actualizar la barra de progreso
                pbar.update(1)

                if not should_continue_analysis():
                    print("Análisis cancelado.")
                    closeCv2Window(cap, out)
                    return

                # Obtén el progreso en porcentaje
                progress_percentage = (pbar.n / total_frames) * 100

                # Calcula el tiempo transcurrido
                elapsed_time = time.time() - start_time

                # Calcula el tiempo restante estimado
                if pbar.n > 0:  # Evita la división por cero
                    avg_time_per_frame = elapsed_time / pbar.n
                    time_remaining = avg_time_per_frame * (total_frames - pbar.n)
                else:
                    time_remaining = 0

                # Convierte el tiempo restante a un formato legible (horas:minutos:segundos)
                hours, remainder = divmod(int(time_remaining), 3600)
                mins, secs = divmod(remainder, 60)
                time_remaining_formatted = f"{hours:02}:{mins:02}:{secs:02}"

                # Imprime los resultados
                print(f"Progreso en variable: {progress_percentage:.2f}%, Tiempo restante en variable: {time_remaining_formatted}")
                if progress_callback:
                    print(f"Calling progress callback with: {progress_percentage}%, Remaining Time: {time_remaining_formatted}")
                    progress_callback(progress_percentage, time_remaining_formatted)

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


    if not should_continue_analysis():
        print("Análisis cancelado.")
        closeCv2Window(cap, out)
        return

    # Post-procesamiento: Generar mapas de trayectoria y guardar datos
    post_processing(keypoint_trajectories,
                    distance_traveled,
                    tiempo_zonas,
                    keypoint_area_data,
                    trajectories_dir,
                    video_name,
                    current_objects, 
                    zona_botellas, 
                    zona_rastis,
                    zona_central,
                    class_colors_matplotlib,
                    id_raton,
                    id_dosis,
                    cantidad,
                    mail_usuario
                    )
    
    