from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
from multiprocessing import Process, Queue
import traceback
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

    # Usar un proceso para manejar el procesamiento del video
    process_queue = Queue()
    process = Process(target=process_video, args=(video_path, process_queue))
    process.start()

    # Iniciar un hilo de fondo para monitorear el progreso
    socketio.start_background_task(target=monitor_progress, process_queue=process_queue)

    return jsonify({"status": "success", "message": "El video se está procesando"}), 200

def process_video(video_path, process_queue):
    try:
        def progress_callback(progress_percentage):
            progress = round(float(progress_percentage), 2)
            process_queue.put(progress)  # Poner el progreso en la cola

        main(
            video_path, 
            os.path.join(os.path.dirname(__file__), '..', 'models', 'keypoint_detection', 'config', 'config.yaml'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'keypoint_detection', 'Bests', 'best_model.pth.tar'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov11_segmentation', 'yolov11x_segmentation', 'weights', 'best.pt'),
            progress_callback
        )
        process_queue.put("done")  # Indicar que el proceso ha terminado
    except Exception as e:
        error_msg = f"Error al procesar el video: {str(e)}\n{traceback.format_exc()}"
        process_queue.put(error_msg)  # Poner el error en la cola

def monitor_progress(process_queue):
    while True:
        try:
            progress = process_queue.get(timeout=5)  # Use timeout to prevent blocking indefinitely
            if progress == "done":
                print("Análisis completado")
                socketio.emit('progress_update', {'progress': 100})
                break
            elif isinstance(progress, str) and progress.startswith("Error"):
                print(progress)
                socketio.emit('error', {'message': 'Error al procesar el video'})
                break
            else:
                print(f"Progreso: {progress}%")
                socketio.emit('progress_update', {'progress': progress})
        except Exception as e:
            print(f"Error monitoreando el progreso: {str(e)}")
            break

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

@socketio.on('test_event')
def handle_test_event(data):
    print('Evento de prueba recibido:', data)
    socketio.emit('test_response', {'response': 'Evento de prueba recibido correctamente'})

if __name__ == "__main__":
    socketio.run(app, debug=True, log_output=True)