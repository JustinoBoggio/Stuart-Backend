import torch
from torch.utils.data import DataLoader
import yaml
from ..models.hrnet import get_pose_net
from ..datasets.coco_dataset import CocoKeypointsDataset
import sys
import os

# Agregar el directorio de modelos a la ruta de Python
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Cargar configuración
with open("hrnet_config.yaml") as f:
    cfg = yaml.safe_load(f)

# Inicializar el modelo
model = get_pose_net(num_keypoints=cfg['MODEL']['NUM_JOINTS'])
device = torch.device("cpu")
model.to(device)

# Cargar el modelo entrenado
checkpoint_path = "path_to_trained_model.pth"
model.load_state_dict(torch.load(checkpoint_path))

# Cargar dataset de validación
val_dataset = CocoKeypointsDataset(cfg['VALIDATION']['ROOT'], cfg['VALIDATION']['ANNOTATIONS'], cfg['DATASET']['IMG_SIZE'])
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Evaluación
model.eval()
total_loss = 0.0
criterion = torch.nn.MSELoss()

with torch.no_grad():
    for images, keypoints in val_loader:
        images = images.to(device)
        keypoints = keypoints.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        total_loss += loss.item()

avg_loss = total_loss / len(val_loader)
print(f"Validation Loss: {avg_loss}")