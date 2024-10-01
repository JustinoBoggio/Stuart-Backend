# import torch
# from torchvision import transforms
# from torchvision.datasets import CocoDetection
# from PIL import Image

# class CocoKeypointsDataset(CocoDetection):
#     def __init__(self, root, annFile, img_size):
#         super(CocoKeypointsDataset, self).__init__(root, annFile)
#         self.img_size = img_size
#         self.transform = transforms.Compose([
#             transforms.Resize(self.img_size),
#             transforms.ToTensor(),
#         ])

#         # Pre-filtrar las imágenes que no tienen keypoints
#         self.valid_indices = self._filter_images_with_keypoints()
    
#     def _filter_images_with_keypoints(self):
#         # Recorremos el dataset para quedarnos solo con los índices válidos
#         valid_indices = []
#         for idx in range(len(self)):
#             _, target = super(CocoKeypointsDataset, self).__getitem__(idx)
#             if len(target) > 0 and 'keypoints' in target[0]:
#                 valid_indices.append(idx)
#         return valid_indices
    
#     def __len__(self):
#         # El tamaño del dataset será el número de imágenes válidas
#         return len(self.valid_indices)
    
# def __getitem__(self, index):
#         # Acceder a las imágenes filtradas mediante los índices válidos
#         real_index = self.valid_indices[index]
#         img, target = super(CocoKeypointsDataset, self).__getitem__(real_index)

#         # Aplicar transformaciones a la imagen
#         img = self.transform(img)

#         # Extraer y procesar los keypoints
#         keypoints = target[0]['keypoints']
#         keypoints_tensor = torch.tensor(keypoints).view(-1, 3)  # Formato [x, y, visibility]

#         # Normalizar los puntos clave al rango [0, 1] en función del tamaño de la imagen
#         keypoints_tensor[:, :2] /= torch.tensor(self.img_size)

#         return img, keypoints_tensor

# import torch
# from torchvision import transforms
# from torchvision.datasets import CocoDetection

# class CocoKeypointsDataset(CocoDetection):
#     def __init__(self, root, annFile, img_size):
#         super(CocoKeypointsDataset, self).__init__(root, annFile)
#         self.img_size = img_size
#         self.transform = transforms.Compose([
#             transforms.Resize(self.img_size),  # Redimensionar las imágenes
#             transforms.ToTensor(),  # Convertir la imagen a tensor
#         ])

#         # Inicializa self.valid_indices antes de llamarlo
#         self.valid_indices = self._filter_images_with_keypoints()

#     def _filter_images_with_keypoints(self):
#         # Recorremos el dataset para quedarnos solo con los índices válidos
#         valid_indices = []
#         for idx in range(super(CocoKeypointsDataset, self).__len__()):
#             _, target = super(CocoKeypointsDataset, self).__getitem__(idx)
#             if len(target) > 0 and 'keypoints' in target[0]:
#                 valid_indices.append(idx)
#         return valid_indices

#     def __len__(self):
#         # El tamaño del dataset será el número de imágenes válidas
#         return len(self.valid_indices)

#     def __getitem__(self, index):
#         # Acceder a las imágenes filtradas mediante los índices válidos
#         real_index = self.valid_indices[index]
#         img, target = super(CocoKeypointsDataset, self).__getitem__(real_index)

#         # Aplicar transformaciones a la imagen
#         img = self.transform(img)

#         # Extraer y procesar los keypoints
#         keypoints = target[0]['keypoints']
#         keypoints_tensor = torch.tensor(keypoints).view(-1, 3)  # Formato [x, y, visibility]

#         # Normalizar los puntos clave al rango [0, 1] en función del tamaño de la imagen
#         keypoints_tensor[:, :2] /= torch.tensor(self.img_size)

#         return img, keypoints_tensor


# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
# import json
# import os
# from PIL import Image
# import numpy as np
# from utils.keypoint_transforms import KeypointTransform, RandomHorizontalFlip, RandomRotation

# class COCODataset(Dataset):
#     def __init__(self, json_file, img_dir, transform=None, is_train=True):
#         with open(json_file) as f:
#             self.data = json.load(f)
#         self.img_dir = img_dir

#         # Utilizar la nueva transformación personalizada para imagen y puntos clave
#         self.transform = transform if transform else KeypointTransform(
#             transforms=[RandomHorizontalFlip(), RandomRotation(30)]
#         )
#         self.is_train = is_train

#     def __len__(self):
#         return len(self.data['images'])

#     def __getitem__(self, idx):
#         img_info = self.data['images'][idx]
#         img_id = img_info['id']
#         img_path = os.path.join(self.img_dir, img_info['file_name'])
#         image = Image.open(img_path).convert('RGB')

#         # Obtener las anotaciones correspondientes
#         keypoints = np.zeros((8, 3))  # 8 puntos (x, y, visible)
#         for ann in self.data['annotations']:
#             if ann['image_id'] == img_id:
#                 keypoints = np.array(ann['keypoints']).reshape(-1, 3)

#         # Convertir los puntos clave a un formato compatible
#         if self.transform:
#             image, keypoints = self.transform(image, keypoints)  # Aplicar la transformación personalizada

#         # Generar los heatmaps de los puntos clave
#         heatmaps = self.generate_heatmaps(keypoints, image.size)

#         return transforms.ToTensor()(image), torch.tensor(heatmaps).float()

#     def generate_heatmaps(self, keypoints, img_size):
#         """Genera mapas de calor basados en los puntos clave."""
#         heatmaps = np.zeros((8, 160, 160), dtype=np.float32)  # Salida de 160x160 para cada punto
#         for idx, (x, y, v) in enumerate(keypoints):
#             if v > 0:  # Punto visible
#                 x = int(x * 160 / img_size[0])  # Ajustar las coordenadas al tamaño del heatmap
#                 y = int(y * 160 / img_size[1])
#                 self.add_gaussian(heatmaps[idx], x, y, sigma=2)
#         return heatmaps

#     @staticmethod
#     def add_gaussian(heatmap, x, y, sigma=2):
#         """Agrega un mapa de calor gaussiano para un punto clave."""
#         temp_size = sigma * 3
#         ul = [int(x - temp_size), int(y - temp_size)]
#         br = [int(x + temp_size + 1), int(y + temp_size + 1)]
#         if ul[0] >= 160 or ul[1] >= 160 or br[0] < 0 or br[1] < 0:
#             return heatmap

#         # Generar la región Gaussiana
#         size = 2 * temp_size + 1
#         x, y = np.meshgrid(np.arange(size), np.arange(size))
#         g = np.exp(-((x - temp_size) ** 2 + (y - temp_size) ** 2) / (2 * sigma ** 2))

#         g_x = max(0, -ul[0]), min(br[0], 160) - ul[0]
#         g_y = max(0, -ul[1]), min(br[1], 160) - ul[1]
#         img_x = max(0, ul[0]), min(br[0], 160)
#         img_y = max(0, ul[1]), min(br[1], 160)

#         heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
#         return heatmap

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
from PIL import Image
import numpy as np
from utils.keypoint_transforms import KeypointTransform, RandomHorizontalFlip, RandomRotation

class COCODataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None, is_train=True):
        with open(json_file) as f:
            self.data = json.load(f)
        self.img_dir = img_dir

        self.transform = transform
        self.is_train = is_train

        # Obtener un mapeo de image_id a anotaciones
        self.annotations = {}
        for ann in self.data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Redimensionar la imagen a 640x640
        original_size = image.size
        image = image.resize((640, 640))

        # Obtener las anotaciones correspondientes
        keypoints = np.zeros((8, 3))  # 8 puntos (x, y, visible)
        if img_id in self.annotations:
            ann = self.annotations[img_id][0]  # Suponiendo una anotación por imagen
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)

            # Escalar los puntos clave al nuevo tamaño de imagen
            scale_x = 640 / original_size[0]
            scale_y = 640 / original_size[1]
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y

        # Aplicar transformaciones
        if self.transform:
            image, keypoints = self.transform(image, keypoints)

        # Generar los mapas de calor de los puntos clave
        heatmaps = self.generate_heatmaps(keypoints, (160, 160))  # Tamaño de los mapas de calor

        # Normalizar la imagen
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = transforms.ToTensor()(image)
        image = normalize(image)

        return image, torch.tensor(heatmaps).float()

    def generate_heatmaps(self, keypoints, heatmap_size):
        """Genera mapas de calor basados en los puntos clave."""
        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros((num_keypoints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
        for idx, (x, y, v) in enumerate(keypoints):
            if v > 0:  # Punto visible
                x = int(x * heatmap_size[0] / 640)  # Ajustar las coordenadas al tamaño del heatmap
                y = int(y * heatmap_size[1] / 640)
                self.add_gaussian(heatmaps[idx], x, y, sigma=2)
        return heatmaps

    @staticmethod
    def add_gaussian(heatmap, x, y, sigma=2):
        """Agrega un mapa de calor gaussiano para un punto clave."""
        tmp_size = sigma * 3
        ul = [int(x - tmp_size), int(y - tmp_size)]
        br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]

        if ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] \
           or br[0] < 0 or br[1] < 0:
            # El gaussiano está fuera del mapa de calor
            return heatmap

        size = 2 * tmp_size + 1
        x_range = np.arange(0, size, 1, np.float32)
        y_range = x_range[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x_range - x0) ** 2 + (y_range - y0) ** 2) / (2 * sigma ** 2))

        # Calcular los límites del gaussiano y del mapa de calor
        g_x = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]
        img_x = max(0, ul[0]), min(br[0], heatmap.shape[1])
        img_y = max(0, ul[1]), min(br[1], heatmap.shape[0])

        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )
        return heatmap