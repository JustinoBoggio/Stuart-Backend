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
#     model = get_pose_net(cfg)
    
#     # Cargar pesos del modelo
#     checkpoint = torch.load(model_path, weights_only=True, map_location='cpu')  # Cargar en CPU primero
#     model.load_state_dict(checkpoint['state_dict'])
#     model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Mover el modelo a GPU si está disponible
#     model.eval()  # Configurar el modelo en modo evaluación
#     return model

# def preprocess_image(image_path):
#     # Cargar la imagen
#     image = Image.open(image_path).convert('RGB')
#     # Redimensionar la imagen a 640x640
#     image = image.resize((640, 640))

#     # Transformar la imagen a tensor
#     image_tensor = transforms.ToTensor()(image)
#     image_tensor = image_tensor.unsqueeze(0)  # Añadir dimensión de batch
#     return image_tensor

# def postprocess_heatmaps(heatmaps):
#     # Convertir los mapas de calor a coordenadas de keypoints
#     keypoints = []
#     num_keypoints = heatmaps.shape[1]
    
#     for i in range(num_keypoints):
#         heatmap = heatmaps[0, i, :, :].detach().cpu().numpy()
#         y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
#         keypoints.append((x, y))  # (x, y) formato de coordenadas
        
#     return keypoints

# def draw_keypoints(image, keypoints):
#     # Dibuja los keypoints y las conexiones
#     for x, y in keypoints:
#         cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Dibuja el keypoint

#     # Conectar keypoints (puedes definir conexiones como desees)
#     connections = [(1, 3), (0, 1), (2, 3), (5, 6), (3, 4), (4, 5), (0, 2), (6, 7)]
    
#     for a, b in connections:
#         cv2.line(image, (keypoints[a][0], keypoints[a][1]), (keypoints[b][0], keypoints[b][1]), (255, 0, 0), 2)

# def main(image_path, config_path, model_path):
#     model = load_model(config_path, model_path)
#     image_tensor = preprocess_image(image_path)

#     # Realizar la inferencia, moviendo la imagen a la GPU
#     with torch.no_grad():
#         outputs = model(image_tensor.to('cuda' if torch.cuda.is_available() else 'cpu'))  # Asegúrate de mover la imagen a la GPU

#     # Procesar los resultados
#     keypoints = postprocess_heatmaps(outputs)

#     # Mostrar la imagen con los keypoints
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (640, 640))  # Asegurarse de que la imagen tenga la misma resolución
#     draw_keypoints(image, keypoints)
    
#     # Mostrar la imagen
#     cv2.imshow('Keypoints', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     image_path = "C:/Users/leone/Desktop/Imagenes/Pruebas/frame_140.jpg"  # Ruta a la imagen de entrada
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
        keypoints.append((int(x), int(y)))
        
    return keypoints

def draw_keypoints(image, keypoints):
    # Dibuja los keypoints y las conexiones
    for x, y in keypoints:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Definir las conexiones entre puntos clave si es necesario
    connections = [(1,3),(0,1),(2,3),(5,6),(3,4),(4,5),(0,2),(6,7)] #[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)] ejemplo
    
    for a, b in connections:
        cv2.line(image, keypoints[a], keypoints[b], (255, 0, 0), 2)

def main(image_path, config_path, model_path):
    model = load_model(config_path, model_path)
    image_tensor = preprocess_image(image_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Realizar la inferencia
    with torch.no_grad():
        outputs = model(image_tensor.to(device))

    # Procesar los resultados
    keypoints = postprocess_heatmaps(outputs)

    # Mostrar la imagen con los keypoints
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    draw_keypoints(image, keypoints)
    
    # Mostrar la imagen
    cv2.imshow('Keypoints', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #image_path = "data/Pruebas/frame_140.jpg"  # Ruta a la imagen de entrada
    #image_path = "data/Pruebas/frame_90.jpg"  # Ruta a la imagen de entrada
    #image_path = "data/Pruebas/frame_95.jpg"  # Ruta a la imagen de entrada
    #image_path = "data/Pruebas/frame_131.jpg"  # Ruta a la imagen de entrada
    image_path = "data/Pruebas/frame_145.jpg"  # Ruta a la imagen de entrada
    #image_path = "data/Pruebas/frame_153.jpg"  # Ruta a la imagen de entrada
    config_path = "config/config.yaml"  # Ruta al archivo de configuración
    model_path = "./outputs/models/best_model.pth.tar"  # Ruta al modelo entrenado

    main(image_path, config_path, model_path)