import eventlet
eventlet.monkey_patch()  # Esto parchea las bibliotecas estándar para que sean compatibles con eventlet

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import time
import pyodbc
from threading import Thread
from process_video_v3 import main, cancel_analysis  # Importa la función main de tu script de procesamiento
from db_connection import get_db_connection
from save_db import get_or_create_raton_id, get_or_create_dosis_id
import bcrypt  # Asegúrate de instalar bcrypt para manejar la verificación de contraseñas


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})
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

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email y contraseña son requeridos'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'No se pudo conectar a la base de datos'}), 500

    cursor = conn.cursor()
    try:
        # Verifica si el correo ya está registrado
        cursor.execute('SELECT COUNT(*) FROM Usuario WHERE mail = ?', (email,))
        if cursor.fetchone()[0] > 0:
            return jsonify({'error': 'El correo ya está registrado'}), 400

        # Encripta la contraseña antes de almacenarla
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Inserta el nuevo usuario
        cursor.execute('INSERT INTO Usuario (mail, contraseña) VALUES (?, ?)', (email, hashed_password.decode('utf-8')))
        conn.commit()
    except pyodbc.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify({'message': 'Usuario registrado con éxito'}), 201

@app.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email y contraseña son requeridos'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'No se pudo conectar a la base de datos'}), 500

    cursor = conn.cursor()
    try:
        # Busca el usuario por correo
        cursor.execute('SELECT contraseña FROM Usuario WHERE mail = ?', (email,))
        user = cursor.fetchone()

        if user is None:
            return jsonify({'error': 'Usuario o contraseña incorrectos'}), 401

        # Verifica la contraseña
        stored_password = user[0]
        if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
            return jsonify({'message': 'Inicio de sesión exitoso'}), 200
        else:
            return jsonify({'error': 'Usuario o contraseña incorrectos'}), 401
    except pyodbc.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()


@app.route('/cancel_analysis', methods=['POST'])
def cancel_analysis_route():
    cancel_analysis()
    return jsonify({'status': 'analysis cancelled'}), 200


@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

if __name__ == "__main__":
    socketio.run(app, debug=True, log_output=True, use_reloader=False)