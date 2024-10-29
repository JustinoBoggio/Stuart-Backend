from ultralytics import YOLO
import cv2
import numpy as np

def load_yolo_model(model_path):
    model = YOLO(model_path, task='segment')
    #model.eval()
    return model

def get_yolo_detections(model, image):
    # Realizar la inferencia
    results = model.predict(
        source=image,
        imgsz=640,
        save=False,
        conf=0.25,
        show=False
    )
    return results

def draw_yolo_detections(image, results, class_colors, target_class_name='BaseCaja'):
    output_image = image.copy()
    masks = results[0].masks
    boxes = results[0].boxes
    
    if len(masks) != len(boxes):
        print("Advertencia: El número de máscaras y cajas detectadas no coincide.")
    
    for mask_idx, mask in enumerate(masks.data):
        try:
            class_id = int(boxes.cls[mask_idx].cpu().numpy())
            class_name = results[0].names[class_id]
        except IndexError:
            print(f"Advertencia: No hay clase asignada para la máscara {mask_idx}.")
            class_name = 'Unknown'
            class_id = -1
        
        color = class_colors.get(class_name, (0, 255, 0))  # Verde por defecto
        
        binary_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            if class_name == target_class_name:
                hull = cv2.convexHull(contour)
                epsilon = 0.02 * cv2.arcLength(hull, True)
                approx_hull = cv2.approxPolyDP(hull, epsilon, True)
                cv2.polylines(output_image, [approx_hull], isClosed=True, color=color, thickness=2)
                # print(f"Aproximación Base Caja: {approx_hull}")
                # print(f"X: {approx_hull[0][0][0]}")
                # print(f"Y: {approx_hull[0][0][1]}")

                # print(f"X,y: {approx_hull[0][0]}")
                # print(f"X,Y: {approx_hull[1][0]}")
                # print(f"X,Y: {approx_hull[2][0]}")
                # print(f"X,Y: {approx_hull[3][0]}")
            else:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.polylines(output_image, [approx], isClosed=True, color=color, thickness=2)
                # print(f"Aproximación Objetos {class_name}: {approx_hull}")
            
            # Agregar etiqueta
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                label = f"{class_name}"
                cv2.putText(output_image, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output_image