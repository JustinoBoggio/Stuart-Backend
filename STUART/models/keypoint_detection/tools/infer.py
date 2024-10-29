# import torch
# import cv2
# import numpy as np
# import yaml
# from models.hrnet import get_pose_net
# from torchvision import transforms
# from PIL import Image

# def load_model(config_path, model_path):
#     # Cargar configuración
#     with open(config_path) as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)
    
#     # Crear el modelo HRNet
#     model = get_pose_net(cfg, is_train=False)
    
#     # Cargar pesos del modelo
#     checkpoint = torch.load(model_path, weights_only=True, map_location='cpu')
#     model.load_state_dict(checkpoint['state_dict'], strict=False)
#     model.to('cuda' if torch.cuda.is_available() else 'cpu')
#     model.eval()
#     return model

# def preprocess_image(image_path):
#     # Cargar la imagen
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize((640, 640))

#     # Transformar la imagen a tensor y normalizar
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#     image_tensor = transform(image)
#     image_tensor = image_tensor.unsqueeze(0)
#     return image_tensor

# def postprocess_heatmaps(heatmaps):
#     # Convertir los mapas de calor a coordenadas de keypoints
#     keypoints = []
#     num_keypoints = heatmaps.shape[1]
#     heatmap_height, heatmap_width = heatmaps.shape[2], heatmaps.shape[3]
    
#     for i in range(num_keypoints):
#         heatmap = heatmaps[0, i, :, :].detach().cpu().numpy()
#         y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
#         # Escalar las coordenadas al tamaño de la imagen original
#         x = x * 640 / heatmap_width
#         y = y * 640 / heatmap_height
#         keypoints.append((int(x), int(y)))
        
#     return keypoints

# def draw_keypoints(image, keypoints):
#     # Dibuja los keypoints y las conexiones
#     for x, y in keypoints:
#         cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

#     # Definir las conexiones entre puntos clave si es necesario
#     connections = [(1,3),(0,1),(2,3),(5,6),(3,4),(4,5),(0,2),(6,7)] #[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)] ejemplo
    
#     for a, b in connections:
#         cv2.line(image, keypoints[a], keypoints[b], (255, 0, 0), 2)

# def main(image_path, config_path, model_path):
#     model = load_model(config_path, model_path)
#     image_tensor = preprocess_image(image_path)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # Realizar la inferencia
#     with torch.no_grad():
#         outputs = model(image_tensor.to(device))

#     # Procesar los resultados
#     keypoints = postprocess_heatmaps(outputs)

#     # Mostrar la imagen con los keypoints
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (640, 640))
#     draw_keypoints(image, keypoints)
    
#     # Mostrar la imagen
#     cv2.imshow('Keypoints', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     image_path = "data/Pruebas/frame_140.jpg"  # Ruta a la imagen de entrada
#     #image_path = "data/Pruebas/frame_90.jpg"  # Ruta a la imagen de entrada
#     #image_path = "data/Pruebas/frame_95.jpg"  # Ruta a la imagen de entrada
#     #image_path = "data/Pruebas/frame_131.jpg"  # Ruta a la imagen de entrada
#     #image_path = "data/Pruebas/frame_145.jpg"  # Ruta a la imagen de entrada
#     #image_path = "data/Pruebas/frame_153.jpg"  # Ruta a la imagen de entrada
#     config_path = "config/config.yaml"  # Ruta al archivo de configuración
#     model_path = "./outputs/models/best_model.pth.tar"  # Ruta al modelo entrenado

#     main(image_path, config_path, model_path)


import torch
import cv2
import numpy as np
import yaml
from models.hrnet import get_pose_net
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_model(config_path, model_path):
    # Cargar configuración
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # Crear el modelo HRNet
    model = get_pose_net(cfg, is_train=False)
    
    # Cargar pesos del modelo
    checkpoint = torch.load(model_path, weights_only=True, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model

def preprocess_image(image_path):
    # Cargar la imagen
    image = Image.open(image_path).convert('RGB')
    image = image.resize((640, 640))

    # Transformar la imagen a tensor y normalizar
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

# Definir los nombres de los keypoints en el orden correcto
keypoint_names = ["Nariz","Oreja Derecha","Oreja Izquierda","Nuca","Columna Media","Base Cola","Mitad Cola","Final Cola"]

def postprocess_heatmaps(heatmaps):
    # Convertir los mapas de calor a coordenadas de keypoints
    keypoints = []
    num_keypoints = heatmaps.shape[1]
    heatmap_height, heatmap_width = heatmaps.shape[2], heatmaps.shape[3]
    
    for i in range(num_keypoints):
        heatmap = heatmaps[0, i, :, :].detach().cpu().numpy()
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        # Escalar las coordenadas al tamaño de la imagen original
        x = x * 640 / heatmap_width
        y = y * 640 / heatmap_height
        # Incluir el nombre del keypoint
        keypoints.append({'name': keypoint_names[i], 'position': (int(x), int(y))})
        
    return keypoints

# Definir las conexiones entre puntos clave usando los nombres
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

def draw_keypoints(image, keypoints):
    # Crear un diccionario para acceder rápidamente a los keypoints por nombre
    keypoints_dict = {kp['name']: kp['position'] for kp in keypoints}
    
    # Dibuja los keypoints y sus nombres
    for kp in keypoints:
        x, y = kp['position']
        name = kp['name']
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        #cv2.putText(image, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Dibuja las conexiones entre los keypoints
    for a_name, b_name in connections:
        if a_name in keypoints_dict and b_name in keypoints_dict:
            pt_a = keypoints_dict[a_name]
            pt_b = keypoints_dict[b_name]
            cv2.line(image, pt_a, pt_b, (255, 0, 0), 2)

def main(image_path, config_path, model_path, visualize_individual=False):
    model = load_model(config_path, model_path)
    image_tensor = preprocess_image(image_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image = cv2.imread(image_path)

    # Realizar la inferencia
    with torch.no_grad():
        outputs = model(image_tensor.to(device))

    # Procesar los resultados
    keypoints = postprocess_heatmaps(outputs)

    if visualize_individual:
        # Visualización individual de keypoints
        visualize_keypoints_individually(image, keypoints)
    else:
        # Mostrar todos los keypoints juntos
        draw_keypoints(image, keypoints)
        # Mostrar la imagen
        cv2.imshow('Keypoints', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def draw_single_keypoint(image, keypoint):
    # Copiar la imagen para no modificar la original
    img_copy = image.copy()
    
    x, y = keypoint['position']
    name = keypoint['name']
    
    # Dibujar el keypoint
    cv2.circle(img_copy, (x, y), 3, (0, 255, 0), -1)
    cv2.putText(img_copy, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img_copy
def visualize_keypoints_individually(image, keypoints):
    # Convertir la imagen de BGR a RGB para matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Crear una figura con subplots
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, kp in enumerate(keypoints):
        # Dibujar solo el keypoint actual
        img_with_kp = draw_single_keypoint(image_rgb, kp)
        
        # Mostrar la imagen en el subplot correspondiente
        axes[idx].imshow(img_with_kp)
        axes[idx].axis('off')
        axes[idx].set_title(kp['name'])
    
    # Ajustar el espaciado entre subplots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #image_path = "data/Pruebas/frame_140.jpg"  # Ruta a la imagen de entrada
    #image_path = "data/Pruebas/frame_90.jpg"  # Ruta a la imagen de entrada
    #image_path = "data/Pruebas/frame_95.jpg"  # Ruta a la imagen de entrada
    #image_path = "data/Pruebas/frame_131.jpg"  # Ruta a la imagen de entrada
    #image_path = "data/Pruebas/frame_145.jpg"  # Ruta a la imagen de entrada
    image_path = "data/Pruebas/frame_153.jpg"  # Ruta a la imagen de entrada
    config_path = "config/config.yaml"  # Ruta al archivo de configuración
    model_path = "./Bests desde 0/150 epoch and sigma 1.5/best_model.pth.tar"  # Ruta al modelo entrenado

    main(image_path, config_path, model_path, False)