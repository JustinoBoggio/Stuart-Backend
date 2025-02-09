from db_connection import get_db_connection

def insert_video(video_name, id_raton, nro_muestra, id_tipo_prueba, id_dosis, cantidad, mail_usuario):
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

        # Insertar el nuevo registro con los parámetros dinámicos
        cursor.execute("""
            INSERT INTO Video (idVideo, idRaton, nroMuestra, idTipoPrueba, idDosis, cantidad, mail_usuario)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (video_name, id_raton, nro_muestra, id_tipo_prueba, id_dosis, cantidad, mail_usuario))
        conn.commit()
        print(f"Video '{video_name}' insertado correctamente.")
    except Exception as e:
        print(f"Error al insertar el video: {e}")
    finally:
        cursor.close()
        conn.close()

def insert_results_to_db(distances, area_central_data, times_data, trajectory_data, video_name, id_raton, id_dosis, nro_muestra, id_tipo_prueba, cantidad, mail_usuario):
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
    insert_video(video_name, id_raton, nro_muestra, id_tipo_prueba, id_dosis, cantidad, mail_usuario)

    conn = get_db_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()

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
            entries = traj.get('entries', 0)
            exits = traj.get('exits', 0)

            if keypoint is None or trajectory_image is None:
                print(f"Advertencia: Datos incompletos para el keypoint {keypoint}.")
                continue

            cursor.execute("""
                INSERT INTO Trayectoria (idVideo, mapaTrayectoria, distanciaRecorrida, descripcion, distanciaRecorrida_AC, nro_entrada_AC, nro_salida_AC)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                id_video,
                trajectory_image,
                distance,
                keypoint,
                area_central_distance,
                entries,
                exits
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