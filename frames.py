import cv2
import os

video_path = 'ruta_al_video.mp4'
output_dir = 'frames/'

# Crear el directorio de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Obtener la tasa de frames por segundo del video
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_rate == 0:  # Extraer un frame por segundo
        frame_filename = os.path.join(output_dir, f'frame_{count}.jpg')
        cv2.imwrite(frame_filename, frame)
    count += 1

cap.release()
print("Extracción de frames completa.")
