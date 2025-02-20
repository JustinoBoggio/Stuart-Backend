from db_connection import get_db_connection

def get_or_create_raton_id(sexo, id_raza):
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()

        # Verificar si el ratón ya existe
        query = """
        SELECT idRaton FROM Raton WHERE sexo = ? AND idRaza = ?
        """
        cursor.execute(query, (sexo, id_raza))
        row = cursor.fetchone()

        if row:
            # Si existe, devuelve el idRaton
            return row[0]

        # Si no existe, inserta un nuevo registro
        insert_query = """
        INSERT INTO Raton (sexo, idRaza) OUTPUT INSERTED.idRaton VALUES (?, ?)
        """
        cursor.execute(insert_query, (sexo, id_raza))
        new_id = cursor.fetchone()[0]
        conn.commit()
        return new_id

    except Exception as e:
        print(f"Error al obtener o crear ratón: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def get_or_create_dosis_id(dose_description):
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        
        # Buscar si la dosis ya existe
        cursor.execute("SELECT idDosis FROM Dosis WHERE descripcion = ?", (dose_description,))
        row = cursor.fetchone()
        
        if row:
            return row[0]
        
        # Si no existe, crear una nueva dosis
        cursor.execute("INSERT INTO Dosis (descripcion) VALUES (?)", (dose_description,))
        conn.commit()
        
        # Obtener el idDosis recién creado
        cursor.execute("SELECT @@IDENTITY")
        new_id = cursor.fetchone()[0]
        return new_id

    except Exception as e:
        print(f"Error al obtener o crear dosis: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def insert_video(video_name, id_raton, id_dosis, cantidad, mail_usuario):
    """Inserta un video en la tabla Video, reemplazando el existente si ya hay uno con el mismo idVideo."""
    conn = get_db_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()

        # Verificar si el idVideo ya existe
        cursor.execute("SELECT COUNT(*) FROM Video WHERE idVideo = ?", (video_name,))
        count = cursor.fetchone()[0]

        if count > 0:
            # Eliminar el registro existente
            cursor.execute("DELETE FROM Video WHERE idVideo = ?", (video_name,))
            conn.commit()
            print(f"Video '{video_name}' existente eliminado.")

        # Antes de llamar a la consulta, verificamos y convertimos la variable
        if cantidad is None or cantidad == '':
            cantidad_real = None
        else:
            cantidad_real = cantidad

        # Insertar el nuevo registro con los parámetros dinámicos
        cursor.execute("""
            INSERT INTO Video (idVideo, idRaton, idDosis, cantidad, mail_usuario)
            VALUES (?, ?, ?, ?, ?)
        """, (video_name, id_raton, id_dosis, cantidad_real, mail_usuario))
        conn.commit()
        print(f"Video '{video_name}' insertado correctamente.")
    except Exception as e:
        print(f"Error al insertar el video: {e}")
    finally:
        cursor.close()
        conn.close()

def insert_results_to_db(distances, area_central_data, times_data, trajectory_data, video_name, id_raton, id_dosis, cantidad, mail_usuario):
    """
    Inserta los resultados del procesamiento en las tablas Trayectoria y TiempoCuriosidad.

    Args:
        distances (dict): Diccionario con las distancias recorridas por los keypoints.
        area_central_data (dict): Diccionario con datos del área central (distancia, entradas, salidas, tiempo).
        times_data (dict): Diccionario con los tiempos en zonas de interés.
        trajectory_data (list): Lista de diccionarios con los mapas de trayectoria como imágenes en bytes.
        video_name (str): Nombre del video procesado.
        id_raton (int): ID del ratón.
        id_dosis (int): ID de la dosis.
        nro_muestra (int): Número de muestra.
        id_tipo_prueba (int): ID del tipo de prueba.
        cantidad (str): Cantidad de dosis.
        mail_usuario (str): Email del usuario.
    """

    insert_video(video_name, id_raton, id_dosis, cantidad, mail_usuario)

    conn = get_db_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()

        # Verificar si el usuario existe
        cursor.execute("SELECT COUNT(*) FROM Usuario WHERE mail = ?", (mail_usuario,))
        user_count = cursor.fetchone()[0]

        if user_count == 0:
            print(f"Error: El correo '{mail_usuario}' no existe en la tabla Usuario.")
            return
        
        # Obtener el idVideo recién insertado
        cursor.execute("SELECT idVideo FROM Video WHERE idVideo = ?", (video_name,))
        video_row = cursor.fetchone()
        if not video_row:
            print(f"Error: No se encontró el idVideo para el video '{video_name}'.")
            return
        id_video = video_row[0]

        # Insertar en la tabla Trayectoria
        for traj in trajectory_data:
            keypoint = traj.get('keypoint', None)
            distance = traj.get('distance', 0.0)
            trajectory_image = traj.get('map', None)
            area_central_distance = traj.get('area_central', 0.0)
            tiempo_area_central = traj.get('time_ac', 0.0)
            entries = traj.get('entries', 0)
            exits = traj.get('exits', 0)

            if keypoint is None or trajectory_image is None:
                print(f"Advertencia: Datos incompletos para el keypoint {keypoint}.")
                continue

            cursor.execute("""
                INSERT INTO Trayectoria (idVideo, mapaTrayectoria, distanciaRecorrida, descripcion, distanciaRecorrida_AC, nro_entrada_AC, nro_salida_AC, tiempo_dentro_AC)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                id_video,
                trajectory_image,
                distance,
                keypoint,
                area_central_distance,
                entries,
                exits,
                tiempo_area_central
            ))
        print("Datos insertados correctamente en la tabla Trayectoria")

        # Insertar en la tabla TiempoCuriosidad
        for objeto_interes, tiempo_curiosidad in times_data.items():
            cursor.execute("""
                INSERT INTO TiempoCuriosidad (idVideo, objetoInteres, tiempoCuriosidad)
                VALUES (?, ?, ?)
            """, (
                id_video,
                objeto_interes,
                tiempo_curiosidad
            ))
        print(f"Datos insertados correctamente en la tabla TiempoCuriosidad para el video '{video_name}'.")

        conn.commit()
        print(f"Resultados insertados correctamente en la base de datos para el video '{video_name}'.")
    except Exception as e:
        print(f"Error al insertar datos en la base de datos: {e}")
    finally:
        cursor.close()
        conn.close()