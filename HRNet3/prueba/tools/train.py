# import os
# import yaml
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from datasets.coco_dataset import COCODataset
# from utils.keypoint_transforms import KeypointTransform, RandomHorizontalFlip, RandomRotation
# from models.hrnet import get_pose_net
# from utils.utils import AverageMeter, save_checkpoint
# from torchvision import transforms
# import argparse

# torch.cuda.empty_cache()
# # Cargar la configuración desde el archivo config.yaml
# parser = argparse.ArgumentParser(description='HRNet Keypoint Training')
# parser.add_argument('--config', default='config/config.yaml', type=str, help='Ruta al archivo de configuración')
# parser.add_argument('--pretrained', default='pose_higher_hrnet_w32_512.pth', type=str, help='Ruta al archivo de pesos preentrenados')
# parser.add_argument('--device', default='cuda', type=str, help='cuda')
# args = parser.parse_args()

# # Verificar si CUDA está disponible
# device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
# print(f'Entrenando en {device}')

# # Cargar configuración desde el archivo config.yaml
# with open(args.config) as f:
#     cfg = yaml.load(f, Loader=yaml.FullLoader)

# # Definir transformaciones de data augmentation utilizando la clase personalizada
# augmentation = KeypointTransform(
#     transforms=[
#         RandomHorizontalFlip(),
#         RandomRotation(cfg['AUGMENTATION']['ROTATION']),
#     ]
# )

# # Crear el dataset y DataLoader para train y validation
# train_dataset = COCODataset(cfg['DATASET']['TRAIN']['ANNOTATIONS'], cfg['DATASET']['TRAIN']['ROOT'], transform=augmentation)
# val_dataset = COCODataset(cfg['DATASET']['VALIDATION']['ANNOTATIONS'], cfg['DATASET']['VALIDATION']['ROOT'])

# train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=False)

# # Crear el modelo HRNet
# model = get_pose_net(cfg)

# # Transferir el modelo a GPU si está disponible
# model = model.to(device)

# # Cargar pesos preentrenados si se especifican
# if args.pretrained:
#     print(f"Cargando pesos preentrenados desde {args.pretrained}")
#     checkpoint = torch.load(args.pretrained, map_location=device, weights_only=True)
#     model.load_state_dict(checkpoint, strict=False)

# # Definir el criterio de pérdida y el optimizador
# criterion = nn.MSELoss()  # Pérdida basada en Mean Squared Error entre los mapas de calor predichos y reales
# optimizer = optim.Adam(model.parameters(), lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])

# # Definir un programador de tasa de aprendizaje
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['TRAIN']['STEP_SIZE'], gamma=cfg['TRAIN']['GAMMA'])

# # Función para el entrenamiento de una época
# def train_epoch(loader, model, criterion, optimizer, epoch):
#     model.train()
#     losses = AverageMeter()

#     for i, (images, heatmaps) in enumerate(loader):
#         images, heatmaps = images.to(device), heatmaps.to(device)  # Transferir imágenes y heatmaps a GPU

#         outputs = model(images)  # Generar los mapas de calor predichos
#         loss = criterion(outputs, heatmaps)  # Calcular la pérdida

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         losses.update(loss.item(), images.size(0))
#         if i % 10 == 0:
#             print(f'Epoch [{epoch+1}], Step [{i}/{len(loader)}], Loss: {losses.val:.4f} (Avg: {losses.avg:.4f})')

#     return losses.avg

# # Función para la validación
# def validate_epoch(loader, model, criterion):
#     model.eval()
#     losses = AverageMeter()

#     with torch.no_grad():
#         for images, heatmaps in loader:
#             images, heatmaps = images.to(device), heatmaps.to(device)  # Transferir imágenes y heatmaps a GPU
#             outputs = model(images)
#             loss = criterion(outputs, heatmaps)
#             losses.update(loss.item(), images.size(0))

#     print(f'Validation Loss: {losses.avg:.4f}')
#     return losses.avg

# # Entrenamiento y validación
# best_loss = float('inf')
# for epoch in range(cfg['TRAIN']['EPOCHS']):
#     train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch)
#     val_loss = validate_epoch(val_loader, model, criterion)

#     # Guardar el mejor modelo
#     is_best = val_loss < best_loss
#     best_loss = min(val_loss, best_loss)
#     save_checkpoint({
#         'epoch': epoch + 1,
#         'state_dict': model.state_dict(),
#         'best_loss': best_loss,
#         'optimizer': optimizer.state_dict(),
#     }, is_best, filepath=cfg['TRAIN']['SAVE_DIR'])

#     # Actualizar el programador de tasa de aprendizaje
#     scheduler.step()

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.coco_dataset import COCODataset
from utils.keypoint_transforms import KeypointTransform, RandomHorizontalFlip, RandomRotation
from models.hrnet import get_pose_net
from utils.utils import AverageMeter, save_checkpoint
import argparse

torch.cuda.empty_cache()

# Argumentos de línea de comandos
parser = argparse.ArgumentParser(description='HRNet Keypoint Training')
parser.add_argument('--config', default='config/config.yaml', type=str, help='Ruta al archivo de configuración')
parser.add_argument('--device', default='cuda', type=str, help='Dispositivo a utilizar (cuda o cpu)')
args = parser.parse_args()

# Verificar si CUDA está disponible
device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
print(f'Entrenando en {device}')

# Cargar configuración
with open(args.config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Definir transformaciones de data augmentation
augmentation = KeypointTransform(
    transforms=[
        RandomHorizontalFlip(),
        RandomRotation(cfg['AUGMENTATION']['ROTATION']),
    ]
)

# Crear el dataset y DataLoader para train y validation
train_dataset = COCODataset(cfg['DATASET']['TRAIN']['ANNOTATIONS'], cfg['DATASET']['TRAIN']['ROOT'], transform=augmentation)
val_dataset = COCODataset(cfg['DATASET']['VALIDATION']['ANNOTATIONS'], cfg['DATASET']['VALIDATION']['ROOT'])

train_loader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], shuffle=False)

# Crear el modelo HRNet
model = get_pose_net(cfg, is_train=True)

# Transferir el modelo al dispositivo
model = model.to(device)

# Definir el criterio de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])

# Definir un programador de tasa de aprendizaje
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['TRAIN']['STEP_SIZE'], gamma=cfg['TRAIN']['GAMMA'])

# Función para el entrenamiento de una época
def train_epoch(loader, model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()

    for i, (images, heatmaps) in enumerate(loader):
        images, heatmaps = images.to(device), heatmaps.to(device)

        outputs = model(images)
        loss = criterion(outputs, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        if i % 10 == 0:
            print(f'Epoch [{epoch+1}], Step [{i}/{len(loader)}], Loss: {losses.val:.4f} (Avg: {losses.avg:.4f})')

    return losses.avg

# Función para la validación
def validate_epoch(loader, model, criterion):
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for images, heatmaps in loader:
            images, heatmaps = images.to(device), heatmaps.to(device)
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            losses.update(loss.item(), images.size(0))

    print(f'Validation Loss: {losses.avg:.4f}')
    return losses.avg

# Entrenamiento y validación
best_loss = float('inf')
for epoch in range(cfg['TRAIN']['EPOCHS']):
    train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch)
    val_loss = validate_epoch(val_loader, model, criterion)

    # Guardar el mejor modelo
    is_best = val_loss < best_loss
    best_loss = min(val_loss, best_loss)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=cfg['TRAIN']['SAVE_DIR'])

    # Actualizar el programador de tasa de aprendizaje
    scheduler.step()