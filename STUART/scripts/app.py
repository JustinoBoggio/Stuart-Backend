import eventlet
eventlet.monkey_patch()  # Esto parchea las bibliotecas estándar para que sean compatibles con eventlet

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import time
from threading import Thread
from process_video_v3 import main  # Importa la función main de tu script de procesamiento


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Define la carpeta de subida con una ruta absoluta desde la raíz
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data', 'videos')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload_video', methods=['POST'])
def upload_video():
    print("Ruta '/upload_video' llamada")
    
    # Obtén el archivo de video
    video = request.files['videoFile']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    # Obtén los demás datos del formulario
    breed = request.form['breed']
    gender = request.form['gender']
    dose = request.form['dose']
    doseAmount = request.form['doseAmount']
    doseList = request.form['doseList']
    
    print("Datos recibidos:")
    print(f"Raza: {breed}")
    print(f"Sexo: {gender}")
    print(f"Tipo de dosis: {dose}")
    print(f"Dosis Cantidad: {doseAmount}")
    print(f"Dosis Nombre: {doseList}")

    # Inicia el procesamiento del video en segundo plano
    socketio.start_background_task(target=process_video, video_path=video_path)

    return jsonify({"status": "success", "message": "El video se está procesando"}), 200

def process_video(video_path):
    try:
        def progress_callback(progress_percentage, remaining_time):
            progress = round(float(progress_percentage), 2)
            with app.app_context():
                socketio.emit('progress_update', {'progress': progress, 'remaining_time': remaining_time})
            time.sleep(0.01)  # Cede el control del CPU brevemente

        main(
            video_path, 
            os.path.join(os.path.dirname(__file__), '..', 'models', 'keypoint_detection', 'config', 'config.yaml'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'keypoint_detection', 'Bests', 'best_model.pth.tar'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov11_segmentation', 'yolov11x_segmentation', 'weights', 'best.pt'),
            progress_callback  # Pasa la función de callback para el progreso
        )
        print("Análisis completado")
    except Exception as e:
        print(f"Error al procesar el video: {str(e)}")
        socketio.emit('error', {'message': 'Error al procesar el video'})

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

if __name__ == "__main__":
    socketio.run(app, debug=True, log_output=True, use_reloader=False)