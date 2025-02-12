import eventlet
eventlet.monkey_patch()  # Esto parchea las bibliotecas estándar para que sean compatibles con eventlet

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import time
import pyodbc
from threading import Thread
from process_video_v3 import main, cancel_analysis  # Importa la función main de tu script de procesamiento
from db_connection import get_db_connection
from save_db import get_or_create_raton_id, get_or_create_dosis_id
from io import BytesIO


app = Flask(__name__)
#CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})

#CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})
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
    breed_id = request.form['breed']
    gender = request.form['gender']
    dose = request.form['dose']
    doseAmount = request.form['doseAmount']
    dose_id = request.form['doseList']
    
    print("Datos recibidos:")
    print(f"Raza Id: {breed_id}")
    print(f"Sexo: {gender}")
    print(f"Tipo de dosis: {dose}")
    print(f"Dosis Cantidad: {doseAmount}")
    print(f"Dosis Id: {dose_id}")
    mail_usuario = "justino.boggio@cerela.com"

    # Obtener o crear idRaton
    id_raton = get_or_create_raton_id(gender, breed_id)
    
    if id_raton is None:
        return jsonify({"status": "error", "message": "No se pudo obtener o crear idRaton"}), 500

    # Verifica si dose_id está vacío y asigna "Sin Dosis" si es necesario
    if not dose_id:
        doseAmount = None
        dose_id = str(get_or_create_dosis_id("Sin Dosis"))
    
    if not dose_id:
        return jsonify({"status": "error", "message": "No se pudo obtener o crear idDosis"}), 500

    # Inicia el procesamiento del video en segundo plano
    print("Antes de procesar")
    print(dose_id)
    print(doseAmount)
    socketio.start_background_task(target=process_video, video_path=video_path, id_raton=id_raton, dose=dose_id, doseAmount=doseAmount, mail_usuario=mail_usuario)

    return jsonify({"status": "success", "message": "El video se está procesando"}), 200

def process_video(video_path, id_raton, dose, doseAmount, mail_usuario):
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
            id_raton,
            dose,
            doseAmount,
            mail_usuario,
            progress_callback
        )
        print("Análisis completado")
    except Exception as e:
        print(f"Error al procesar el video: {str(e)}")
        socketio.emit('error', {'message': 'Error al procesar el video'})

@app.route('/api/breeds', methods=['GET'])
def get_breeds():
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'No se pudo conectar a la base de datos'}), 500

    cursor = conn.cursor()
    cursor.execute('SELECT idRaza, nombreRaza FROM Raza')
    breeds = cursor.fetchall()
    cursor.close()
    conn.close()

    breeds_list = [{'idRaza': breed[0], 'nombreRaza': breed[1]} for breed in breeds]
    return jsonify(breeds_list)

@app.route('/api/doses', methods=['GET'])
def get_doses():
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'No se pudo conectar a la base de datos'}), 500

    cursor = conn.cursor()
    cursor.execute('SELECT idDosis, descripcion FROM Dosis')
    doses = cursor.fetchall()
    cursor.close()
    conn.close()

    doses_list = [{'idDosis': dose[0], 'descripcion': dose[1]} for dose in doses]
    return jsonify(doses_list)

@app.route('/api/add_breed', methods=['POST'])
def add_breed():
    data = request.json
    nombre_raza = data.get('name')

    if not nombre_raza:
        return jsonify({'error': 'El nombre es obligatorio'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'No se pudo conectar a la base de datos'}), 500

    cursor = conn.cursor()
    try:
        # Insertar sin OUTPUT INSERTED, ya que no necesitamos el ID
        cursor.execute('INSERT INTO Raza (nombreRaza) VALUES (?)', (nombre_raza,))
        conn.commit()
    except pyodbc.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify({'message': f'Raza "{nombre_raza}" agregada con éxito.'}), 201

@app.route('/api/add_dose', methods=['POST'])
def add_dose():
    data = request.json
    descripcion = data.get('name')

    if not descripcion:
        return jsonify({'error': 'El nombre es obligatorio'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'No se pudo conectar a la base de datos'}), 500

    cursor = conn.cursor()
    try:
        # Insertar sin OUTPUT INSERTED, ya que no necesitamos el ID
        cursor.execute('INSERT INTO Dosis (descripcion) VALUES (?)', (descripcion,))
        conn.commit()
    except pyodbc.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify({'message': f'Dosis "{descripcion}" agregada con éxito.'}), 201

@app.route('/cancel_analysis', methods=['POST'])
def cancel_analysis_route():
    cancel_analysis()
    return jsonify({'status': 'analysis cancelled'}), 200


@app.route('/get_videos', methods=['GET'])
def get_videos():
    """
    Obtiene la lista de videos procesados desde la base de datos.
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                v.idVideo, 
                r.sexo, 
                rz.nombreRaza, 
                d.descripcion 
            FROM Video v
            JOIN Raton r ON v.idRaton = r.idRaton
            JOIN Raza rz ON r.idRaza = rz.idRaza
            LEFT JOIN Dosis d ON v.idDosis = d.idDosis
        """)
        videos = cursor.fetchall()

        video_list = []
        for video in videos:
            video_list.append({
                "name": video[0],   # ID del video
                "sex": video[1],    # Sexo del ratón
                "race": video[2],   # Nombre de la raza
                "dose": video[3] if video[3] else "Sin Dosis"  # Nombre de la sustancia o "Sin Dosis"
            })

        return jsonify(video_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/get_results/<video_name>/<keypoint>', methods=['GET'])
def get_results_by_keypoint(video_name, keypoint):
    """
    Obtiene los resultados de un video en particular.
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

    try:
        cursor = conn.cursor()
        
        # Obtener métricas generales filtradas por descripción (keypoint)
        cursor.execute("""
            SELECT distanciaRecorrida, distanciaRecorrida_AC, nro_entrada_AC, nro_salida_AC, tiempo_dentro_AC
            FROM Trayectoria WHERE idVideo = ? AND descripcion = ?
        """, (video_name, keypoint))
        trayectoria = cursor.fetchone()

        # DEBUG: Ver qué devuelve la base de datos
        print(f"Datos de trayectoria para {video_name}: {trayectoria}")

        # Obtener tiempos de curiosidad
        cursor.execute("""
            SELECT objetoInteres, tiempoCuriosidad 
            FROM TiempoCuriosidad WHERE idVideo = ?
        """, (video_name,))
        tiempos = cursor.fetchall()

        # DEBUG: Ver qué tiempos devuelve la base de datos
        print(f"Tiempos de curiosidad para {video_name}: {tiempos}")

        tiempos_data = [{"object": t[0], "time": t[1]} for t in tiempos]

        result_data = {
            "distance": trayectoria[0] if trayectoria else 0.0,  # distanciaRecorrida
            "central_distance": trayectoria[1] if trayectoria else 0.0,  # distanciaRecorrida_AC
            "central_entries": trayectoria[2] if trayectoria else 0,  # nro_entrada_AC
            "central_exits": trayectoria[3] if trayectoria else 0,  # nro_salida_AC
            "central_time": trayectoria[4] if trayectoria else 0.0,  # tiempo_dentro_AC
            "times": tiempos_data
        }

        return jsonify(result_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()


@app.route('/get_trayectoria_image/<video_name>/<keypoint>', methods=['GET'])
def get_trayectoria_image(video_name, keypoint):
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT mapaTrayectoria FROM Trayectoria 
            WHERE idVideo = ? AND descripcion = ?
        """, (video_name, keypoint))
        row = cursor.fetchone()

        if not row or not row[0]:
            return jsonify({"error": "Imagen no encontrada"}), 404

        image_binary = row[0]
        image_stream = BytesIO(image_binary)

        return send_file(image_stream, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/delete_video/<video_name>', methods=['DELETE'])
def delete_video(video_name):
    """
    Elimina un video de la base de datos y su archivo asociado.
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

    try:
        cursor = conn.cursor()

        # Obtener la ruta del video antes de eliminarlo
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)

        # Verificar si el video existe en la base de datos
        cursor.execute("SELECT idVideo FROM Video WHERE idVideo = ?", (video_name,))
        row = cursor.fetchone()

        if not row:
            return jsonify({"error": "El video no existe en la base de datos"}), 404

        # Eliminar registros en la base de datos (referencias en otras tablas)
        cursor.execute("DELETE FROM Trayectoria WHERE idVideo = ?", (video_name,))
        cursor.execute("DELETE FROM TiempoCuriosidad WHERE idVideo = ?", (video_name,))
        cursor.execute("DELETE FROM Video WHERE idVideo = ?", (video_name,))
        conn.commit()

        # Eliminar el archivo de video si existe
        if os.path.exists(video_path):
            os.remove(video_path)

        return jsonify({"status": "success", "message": "Video eliminado correctamente"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()


@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

if __name__ == "__main__":
    socketio.run(app, debug=True, log_output=True, use_reloader=False)