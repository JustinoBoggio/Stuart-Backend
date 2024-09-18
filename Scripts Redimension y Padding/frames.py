import cv2
import os

def extract_frames(video_path, output_folder, time_interval=2):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capturar el video
    vidcap = cv2.VideoCapture(video_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))  # Obtener la tasa de fotogramas del video
    interval_frames = int(fps * time_interval)  # Intervalo en fotogramas

    success, image = vidcap.read()
    count = 0
    saved_count = 0

    while success:
        if count % interval_frames == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{saved_count}.jpg"), image)
            saved_count += 1
        success, image = vidcap.read()
        count += 1

    print(f"Fotogramas extraídos: {saved_count}")

# Ejemplo de uso
#extract_frames("D:/Justino/Tesis/NOR hembras escopolamina completo/G1/Reconocimiento_1A.mp4", "D:/Justino/Tesis/Stuart-Backend/Frames/Reconocimiento_1A", time_interval=2)
extract_frames("D:/Leonel/ISI/Tesis/OneDrive_2024-05-23/NOR hembras escopolamina completo/G2/Test_2_H.mp4", "D:/Leonel/Tesis/Stuart-Backend/Frames/Test_2_H", time_interval=2)