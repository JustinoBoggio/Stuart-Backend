import eventlet
eventlet.monkey_patch()  # Esto parchea las bibliotecas estándar para que sean compatibles con eventlet

from flask import Flask, request, jsonify, send_file, session
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
from io import BytesIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import random
from datetime import datetime
import base64


app = Flask(__name__)
#secret_key = os.urandom(24)
app.secret_key = 'stuart2025' # La clave secreta se usa para firmar las cookies de sesión, de tal manera que si un atacante intenta modificar la cookie, Flask podrá detectarlo y rechazar la sesión.
app.config.update(
    #SESSION_COOKIE_SAMESITE='None',  # Permitir cookies en solicitudes de origen cruzado
    SESSION_COOKIE_SECURE=False,     # No requiere HTTPS en desarrollo; asegúrate de cambiar esto en producción
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'     # Protege las cookies de ser accedidas por JavaScript
)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

socketio = SocketIO(app, cors_allowed_origins="*", manage_session = True)

# Define la carpeta de subida con una ruta absoluta desde la raíz
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'data', 'videos')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload_video', methods=['POST'])
def upload_video():
    print("Ruta '/upload_video' llamada")
    print(f"Contenido de la sesión: {dict(session)}")
    
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
    mail_usuario = session.get('user_email')
    print(f"Mail Usuario: {mail_usuario}")

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
    cursor.execute('SELECT idDosis, descripcion FROM Dosis WHERE descripcion != \'Sin Dosis\' ')
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
            session['user_email'] = email
            print(f"Email almacenado en sesión: {session['user_email']}")
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
                d.descripcion,
                v.cantidad
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
                "dose": video[3], #if video[3] else "Sin Dosis"  # Nombre de la sustancia o "Sin Dosis"
                "amount": video[4]
            })

        return jsonify(video_list)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        conn.close()

@app.route('/get_information/<video_name>', methods=['GET'])
def get_information_by_video_name(video_name):
    """
    Obtiene la informacion de raza, dosis y cantidad de dosis aplicada, mail usuario y fecha y hora.
    """
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

    print(f"Contenido de la sesión: {dict(session)}")
    
    mail_usuario = session.get('user_email')
    now = datetime.now()
    # Formatea la fecha (YYYY-MM-DD) y hora (HH:MM:SS)
    fecha_generacion = now.strftime("%d/%m/%Y")
    hora_generacion = now.strftime("%H:%M:%S")

    try:
        cursor = conn.cursor()
        
        # Obtener métricas generales filtradas por descripción (keypoint)
        cursor.execute("""
            SELECT rz.nombreRaza, r.sexo, d.descripcion, v.cantidad
                FROM Video AS v 
                INNER JOIN Raton AS r ON  r.idRaton = v.idRaton
                INNER JOIN Raza AS rz ON rz.idRaza = r.idRaza
                INNER JOIN Dosis AS d ON d.idDosis = v.idDosis 
            WHERE v.idVideo = ?
        """, (video_name))
        info = cursor.fetchone()

        # DEBUG: Ver qué devuelve la base de datos
        print(f"Información del video {video_name}")

        result_data = {
            "raza": info[0] if info else '',
            "sexo": info[1] if info else '',
            "dosis": info[2] if info else '', 
            "cantidad": info[3] if info else '',
            "usuario":  mail_usuario,
            "fecha": fecha_generacion,
            "hora": hora_generacion
        }

        return jsonify(result_data)

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

@app.route('/get_trayectoria_image_base64/<video_name>/<keypoint>', methods=['GET'])
def get_trayectoria_image_base64(video_name, keypoint):
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

        image_binary = row[0]  # Bytes de la imagen
        # Convertir a base64
        image_base64 = base64.b64encode(image_binary).decode('utf-8')

        return jsonify({"image_base64": image_base64})

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

#################################  Sección Recuperación de Contraseña #################################

# 1) Ruta para verificar existencia de mail
@app.route('/api/checkEmail', methods=['POST'])
def check_email():
    data = request.json
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Debe proporcionar un correo'}), 400

    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'No se pudo conectar a la base de datos'}), 500

    cursor = conn.cursor()
    try:
        cursor.execute('SELECT mail FROM Usuario WHERE mail = ?', (email,))
        user = cursor.fetchone()

        if user is None:
            # No existe
            return jsonify({'error': 'El correo no existe'}), 404
        else:
            # Existe
            return jsonify({'message': 'El correo existe'}), 200

    except pyodbc.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

# 2) Ruta para enviar correo con el código (recibe email y code)
@app.route('/api/sendMail', methods=['POST'])
def send_mail():
    data = request.json
    email = data.get('email')
    code = data.get('code')

    if not email or not code:
        return jsonify({'error': 'Faltan datos: email o code'}), 400

    # Enviar el correo
    if enviar_correo_recovery(email, code):
        return jsonify({'message': 'Correo enviado con éxito'}), 200
    else:
        return jsonify({'error': 'Error al enviar el correo'}), 500

def enviar_correo_recovery(destinatario, code):
    """
    Envía un correo HTML usando la cuenta de Gmail 'stuart.ia.73@gmail.com'.
    Retorna True si se envió con éxito, False si hubo error.
    """
    try:

        # Obtener la fecha y hora actual
        fecha_hora_actual = datetime.now()

        # Formatear la fecha y hora
        formato = fecha_hora_actual.strftime("%d-%m-%Y %H:%M:%S")

        # Datos del remitente
        gmail_user = 'stuart.ia.73@gmail.com'
        gmail_password = 'asmi ssqw lvip fcxn'  # Código de Aplicación PC Leonel

        # Crear mensaje
        mensaje = MIMEMultipart('alternative')
        mensaje['From'] = gmail_user
        mensaje['To'] = destinatario
        mensaje['Subject'] = "STUART - RECUPERACIÓN DE CONTRASEÑA"

        # Cuerpo en HTML
        html_content = f"""
        <html>
        <body style="background-color: #FFFFFF; font-family: Arial, Helvetica, Serif; font-size: 10pt; color: #808083;">
        
            <!-- Encabezado del correo -->
        <h1 style="
            background-color: #FFFFFF;
            margin: 0;
            padding-top: 1em;
            text-align: center;
            color: #FFFFFF;
            font-family: Arial, sans-serif;
            font-size: 16pt;
        ">
        <!-- LOGO CENTRADO SIN FONDO -->
        <img 
            src="cid:stuart_logo"
            alt="Stuart Logo"
            style="width: 250px; height: auto; border: none; display: block; margin: 0 auto;"
        >
        </h1>

            <div style="margin-left: 1%;"> <br> <br>
                <h2 style="background-color: #FFFFFF; margin: 0px; color: #3347ff; font-family: Arial, Helvetica, Serif; font-size: 12pt; padding: 0.1em; white-space: nowrap;"><b>CÓDIGO PARA RESTABLECIMIENTO DE CONTRASEÑA</b></h2> <br>
                <p style="color: black; font-family: Arial, Helvetica, Serif; font-size: 11pt;">Usuario Solicitante: <b>{destinatario}</b></p>
                <p style="color: black; font-family: Arial, Helvetica, Serif; font-size: 11pt;"> Fecha y Hora: <b>{formato}</b></p> </br>
            </div>

            <div style="background-color: #e6e8ff; margin: 1%;">
                <div style="text-align: center;"><br>
                    <p style="color: #3347ff; font-family: Arial, Helvetica, Serif; font-size: 14pt;"><b>CÓDIGO</b></p>
                </div>
                <div style="background-color: #3347ff; color: #FFFFFF; padding: 1%; margin: 1% 25%; margin-bottom: 1%; border-radius: 20px; text-align: center; overflow-x: auto;">
                    <p style="color: White; font-family: Arial, Helvetica, Serif; font-size: 16pt;"><b>{code}</b></p>
                </div> <br>
            </div>
            <div style="background-color: #3347ff; color: #FFFFFF; font-family: Arial, sans-serif; font-size: 11pt; text-align: center; padding: 2% 1% 2% 1%; position: fixed; left: 0; bottom: 0; margin: 1%; margin-left: 0.6%; width: 97%;">
                <!-- unique-id: {code}... -->
                <div> <b>Este código debe ser resguardado y tratado con discreción.</b> </div>
            </div>
        </body>

        </html>
        """

        parte_html = MIMEText(html_content, 'html')
        mensaje.attach(parte_html)

        # Adjuntar la imagen
        with open("./scripts/image/logoStuart.png", "rb") as img:
            mime_img = MIMEImage(img.read(), name="logoStuart.png")
            mime_img.add_header("Content-ID", "<stuart_logo>")  # CID debe estar entre <>
            mime_img.add_header("Content-Disposition", "inline", filename="logoStuart.png")
            mensaje.attach(mime_img)

        # Enviar usando SMTP de Gmail
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(gmail_user, gmail_password)
        servidor.sendmail(gmail_user, destinatario, mensaje.as_string())
        servidor.quit()
        return True

    except Exception as e:
        print("Error al enviar correo:", e)
        return False
    
@app.route('/api/UpdatePassword', methods=['POST'])
def UpdatePassword_user():
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
        # Encripta la contraseña antes de almacenarla
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Inserta el nuevo usuario
        cursor.execute('UPDATE Usuario SET contraseña = ? WHERE mail = ? ;', (hashed_password.decode('utf-8'), email))
        conn.commit()
    except pyodbc.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

    return jsonify({'message': 'Contraseña actualizada correctamente'}), 201


@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

if __name__ == "__main__":
    socketio.run(app, debug=True, log_output=True, use_reloader=False)