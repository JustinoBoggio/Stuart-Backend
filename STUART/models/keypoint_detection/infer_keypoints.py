import torch
import cv2
import numpy as np
import yaml
from models.keypoint_detection.models.hrnet import get_pose_net
from torchvision import transforms
from PIL import Image

def load_keypoint_model(config_path, model_path):
    # Cargar configuración
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # Crear el modelo HRNet
    model = get_pose_net(cfg, is_train=False)
    
    # Cargar pesos del modelo
    checkpoint = torch.load(model_path, weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    return model, device

def preprocess_keypoint_image(image, size=(640, 640)):
    # Convertir imagen de OpenCV (BGR) a PIL (RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).convert('RGB')
    image_pil = image_pil.resize(size)
    
    # Transformar la imagen a tensor y normalizar
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_pil)
    image_tensor = image_tensor.unsqueeze(0)  # Añadir dimensión de batch
    
    # Verificar el rango y canales
    #print(f"Preprocesamiento: Imagen tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
    return image_tensor

def postprocess_keypoints(heatmaps, image_shape=(640, 640)):
    keypoint_names = ["Nariz","Oreja Derecha","Oreja Izquierda","Nuca","Columna Media","Base Cola","Mitad Cola","Final Cola"]
    keypoints = []
    num_keypoints = heatmaps.shape[1]
    heatmap_height, heatmap_width = heatmaps.shape[2], heatmaps.shape[3]
    
    for i in range(num_keypoints):
        heatmap = heatmaps[0, i, :, :].detach().cpu().numpy()
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # Escalar las coordenadas al tamaño de la imagen original
        x = x * image_shape[1] / heatmap_width
        y = y * image_shape[0] / heatmap_height
        # Incluir el nombre del keypoint
        keypoints.append({'name': keypoint_names[i], 'position': (int(x), int(y))})
        
    return keypoints

def draw_keypoints(image, keypoints, connections=None):
    if connections is None:
        connections = [
            ('Nariz', 'Oreja Derecha'),
            ('Nariz', 'Oreja Izquierda'),
            ('Oreja Derecha', 'Nuca'),
            ('Oreja Izquierda', 'Nuca'),
            ('Nuca', 'Columna Media'),
            ('Columna Media', 'Base Cola'),
            ('Base Cola', 'Mitad Cola'),
            ('Mitad Cola', 'Final Cola'),
        ]
    
    # Crear un diccionario para acceder rápidamente a los keypoints por nombre
    keypoints_dict = {kp['name']: kp['position'] for kp in keypoints}
    
    # Dibuja los keypoints y sus nombres
    for kp in keypoints:
        x, y = kp['position']
        name = kp['name']
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        #cv2.putText(image, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Dibuja las conexiones entre los keypoints
    for a_name, b_name in connections:
        if a_name in keypoints_dict and b_name in keypoints_dict:
            pt_a = keypoints_dict[a_name]
            pt_b = keypoints_dict[b_name]
            cv2.line(image, pt_a, pt_b, (255, 0, 0), 2)

def get_keypoints(model, device, image):
    # Preprocesar imagen
    image_tensor = preprocess_keypoint_image(image)
    image_tensor = image_tensor.to(device)
    
    # Realizar la inferencia
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Procesar los resultados
    keypoints = postprocess_keypoints(outputs)
    return keypoints