import os
import torch
import shutil

class AverageMeter:
    """Clase para almacenar y actualizar los valores promedio y actuales de métricas."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Reiniciar todos los contadores."""
        self.val = 0  # Valor actual
        self.avg = 0  # Promedio
        self.sum = 0  # Suma acumulada
        self.count = 0  # Conteo de valores

    def update(self, val, n=1):
        """Actualizar el contador con un nuevo valor."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def save_checkpoint(state, is_best, filepath='checkpoint.pth'):
#     """
#     Guardar el checkpoint del modelo.
#     Args:
#         state: Diccionario con el estado actual del modelo y el optimizador.
#         is_best: Si es el mejor modelo hasta ahora, también guardar como 'best_model.pth'.
#         filepath: Directorio donde se guardará el checkpoint.
#     """
#     if not os.path.exists(filepath):
#         os.makedirs(filepath)

#     filename = os.path.join(filepath, 'checkpoint.pth.tar')
#     try:
#         torch.save(state, filename)
#         if is_best:
#             best_filename = os.path.join(filepath, 'best_model.pth.tar')
#             torch.save(state, best_filename)
#     except Exception as e:
#         print(f"Error al guardar el checkpoint: {e}")
#         shutil.copyfile(filename, best_filename)

def save_checkpoint(state, is_best, filepath):
    """
    Guarda el modelo en el directorio especificado.
    Args:
        state: Diccionario con el estado actual del modelo y el optimizador.
        filepath: Directorio donde se guardará el modelo.
    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    if is_best:
        best_filename = os.path.join(filepath, 'best_model.pth.tar')
        try:
            torch.save(state, best_filename)
            print(f"Mejor modelo guardado en '{best_filename}'")
        except Exception as e:
            print(f"Error al guardar el mejor modelo: {e}")