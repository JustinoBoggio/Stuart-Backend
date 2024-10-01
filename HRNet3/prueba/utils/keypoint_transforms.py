# import numpy as np
# import cv2
# import torch
# import random
# import math
# import torchvision.transforms.functional as F

# class KeypointTransform:
#     """Aplica transformaciones a la imagen y ajusta los puntos clave de manera correspondiente."""
#     def __init__(self, transforms=None):
#         self.transforms = transforms

#     def __call__(self, image, keypoints):
#         """
#         Aplica las transformaciones tanto a la imagen como a los puntos clave.
#         Args:
#             image: PIL Image
#             keypoints: numpy array de forma (N, 3) con (x, y, visibilidad)
#         """
#         # Aplicar las transformaciones si están definidas
#         if self.transforms is not None:
#             for transform in self.transforms:
#                 image, keypoints = transform(image, keypoints)
#         return image, keypoints


# class RandomHorizontalFlip:
#     """Transformación de flip horizontal para la imagen y puntos clave."""
#     def __init__(self, flip_prob=0.5):
#         self.flip_prob = flip_prob

#     def __call__(self, image, keypoints):
#         if random.random() < self.flip_prob:
#             image = F.hflip(image)
#             # Reflejar horizontalmente las coordenadas x de los puntos clave
#             keypoints[:, 0] = image.width - keypoints[:, 0]
#         return image, keypoints


# class RandomRotation:
#     """Transformación de rotación para la imagen y puntos clave."""
#     def __init__(self, degrees):
#         self.degrees = degrees

#     def __call__(self, image, keypoints):
#         angle = random.uniform(-self.degrees, self.degrees)
#         image = F.rotate(image, angle)
        
#         # Ajustar las coordenadas de los puntos clave para la rotación
#         cx, cy = image.width / 2, image.height / 2
#         radians = math.radians(angle)  # Convertir el ángulo a radianes

#         # Copiar las coordenadas de los puntos clave
#         new_keypoints = keypoints.copy()

#         for i, (x, y, v) in enumerate(keypoints):
#             if v > 0:  # Si el punto es visible
#                 # Rotar el punto alrededor del centro usando trigonometría básica
#                 new_x = cx + (x - cx) * math.cos(radians) - (y - cy) * math.sin(radians)
#                 new_y = cy + (x - cx) * math.sin(radians) + (y - cy) * math.cos(radians)
#                 new_keypoints[i, 0] = new_x
#                 new_keypoints[i, 1] = new_y

#         return image, new_keypoints

import numpy as np
import random
import math
import torchvision.transforms.functional as F

class KeypointTransform:
    """Aplica transformaciones a la imagen y ajusta los puntos clave de manera correspondiente."""
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, image, keypoints):
        if self.transforms is not None:
            for transform in self.transforms:
                image, keypoints = transform(image, keypoints)
        return image, keypoints

class RandomHorizontalFlip:
    """Transformación de flip horizontal para la imagen y puntos clave."""
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, keypoints):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            keypoints[:, 0] = image.width - keypoints[:, 0]
            # Opcional: Reordenar los puntos clave si es necesario
        return image, keypoints

class RandomRotation:
    """Transformación de rotación para la imagen y puntos clave."""
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, keypoints):
        angle = random.uniform(-self.degrees, self.degrees)
        image = F.rotate(image, angle)

        # Ajustar las coordenadas de los puntos clave para la rotación
        cx, cy = image.width / 2, image.height / 2
        radians = math.radians(-angle)  # Negativo para rotación inversa

        new_keypoints = keypoints.copy()

        for i, (x, y, v) in enumerate(keypoints):
            if v > 0:
                dx = x - cx
                dy = y - cy
                new_x = dx * math.cos(radians) - dy * math.sin(radians) + cx
                new_y = dx * math.sin(radians) + dy * math.cos(radians) + cy
                new_keypoints[i, 0] = new_x
                new_keypoints[i, 1] = new_y

        return image, new_keypoints