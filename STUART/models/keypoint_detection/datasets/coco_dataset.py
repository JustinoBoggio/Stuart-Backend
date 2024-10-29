import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
from PIL import Image
import numpy as np
from utils.keypoint_transforms import KeypointTransform, RandomHorizontalFlip, RandomRotation

class COCODataset(Dataset):
    def __init__(self, json_file, img_dir, cfg, transform=None, is_train=True):
        with open(json_file) as f:
            self.data = json.load(f)
        self.img_dir = img_dir

        self.transform = transform
        self.is_train = is_train
        self.cfg = cfg

        # Obtener un mapeo de image_id a anotaciones
        self.annotations = {}
        for ann in self.data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # Obtener el valor de sigma desde la configuración
        self.sigma = self.cfg['MODEL']['SIGMA']
        self.image_size = self.cfg['MODEL']['IMAGE_SIZE']
        self.heatmap_size = self.cfg['MODEL']['HEATMAP_SIZE']

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        # Redimensionar la imagen a 640x640
        original_size = image.size
        image = image.resize((self.image_size[0], self.image_size[1]))

        # Obtener las anotaciones correspondientes
        keypoints = np.zeros((self.cfg['MODEL']['NUM_JOINTS'], 3))  # (x, y, visible)
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
        heatmaps = self.generate_heatmaps(keypoints, self.heatmap_size, self.sigma)  # Tamaño de los mapas de calor

        # Normalizar la imagen
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = transforms.ToTensor()(image)
        image = normalize(image)

        return image, torch.tensor(heatmaps).float()

    def generate_heatmaps(self, keypoints, heatmap_size, sigma=2):
        """Genera mapas de calor basados en los puntos clave."""
        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros((num_keypoints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
        for idx, (x, y, v) in enumerate(keypoints):
            if v > 0:  # Punto visible
                x = int(x * heatmap_size[0] / 640)  # Ajustar las coordenadas al tamaño del heatmap
                y = int(y * heatmap_size[1] / 640)
                self.add_gaussian(heatmaps[idx], x, y, sigma)
        return heatmaps
    
    @staticmethod
    def add_gaussian(heatmap, x, y, sigma):
        """Agrega un mapa de calor gaussiano para un punto clave."""
        tmp_size = sigma * 3
        mu_x = int(x + 0.5)
        mu_y = int(y + 0.5)
        w, h = heatmap.shape[1], heatmap.shape[0]
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

        if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
            # El gaussiano está fuera del mapa de calor
            return heatmap

        size = 2 * tmp_size + 1
        x_range = np.arange(0, size, 1, np.float32)
        y_range = x_range[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x_range - x0) ** 2 + (y_range - y0) ** 2) / (2 * sigma ** 2))

        # Calcular los límites del gaussiano y del mapa de calor
        g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
        img_x = max(0, ul[0]), min(br[0], w)
        img_y = max(0, ul[1]), min(br[1], h)

        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
            heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        )
        return heatmap

    # @staticmethod
    # def add_gaussian(heatmap, x, y, sigma=2):
    #     """Agrega un mapa de calor gaussiano para un punto clave."""
    #     tmp_size = sigma * 3
    #     ul = [int(x - tmp_size), int(y - tmp_size)]
    #     br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]

    #     if ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] \
    #        or br[0] < 0 or br[1] < 0:
    #         # El gaussiano está fuera del mapa de calor
    #         return heatmap

    #     size = 2 * tmp_size + 1
    #     x_range = np.arange(0, size, 1, np.float32)
    #     y_range = x_range[:, np.newaxis]
    #     x0 = y0 = size // 2
    #     g = np.exp(- ((x_range - x0) ** 2 + (y_range - y0) ** 2) / (2 * sigma ** 2))

    #     # Calcular los límites del gaussiano y del mapa de calor
    #     g_x = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
    #     g_y = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]
    #     img_x = max(0, ul[0]), min(br[0], heatmap.shape[1])
    #     img_y = max(0, ul[1]), min(br[1], heatmap.shape[0])

    #     heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
    #         heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
    #         g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    #     )
    #     return heatmap