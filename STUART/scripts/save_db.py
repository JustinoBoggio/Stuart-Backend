from db_connection import get_db_connection

def insert_raza():
    """Inserta datos de prueba en la tabla Raza si no existen."""
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Raza")
        count = cursor.fetchone()[0]
        if count == 0:  # Solo insertar si la tabla está vacía
            cursor.executemany("""
                INSERT INTO Raza (nombreRaza) VALUES (?)
            """, [('Sprague Dawley',), ('Wistar',), ('Long-Evans',)])
            conn.commit()
            print("Datos insertados en la tabla Raza")
    except Exception as e:
        print(f"Error al insertar en Raza: {e}")
    finally:
        cursor.close()
        conn.close()


def insert_raton():
    """Inserta datos de prueba en la tabla Raton si no existen."""
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Raton")
        count = cursor.fetchone()[0]
        if count == 0:
            cursor.executemany("""
                INSERT INTO Raton (sexo, idRaza) VALUES (?, ?)
            """, [('M', 1), ('F', 2), ('M', 3)])
            conn.commit()
            print("Datos insertados en la tabla Raton")
    except Exception as e:
        print(f"Error al insertar en Raton: {e}")
    finally:
        cursor.close()
        conn.close()


def insert_dosis():
    """Inserta datos de prueba en la tabla Dosis si no existen."""
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Dosis")
        count = cursor.fetchone()[0]
        if count == 0:
            cursor.executemany("""
                INSERT INTO Dosis (cantidad, sustancia) VALUES (?, ?)
            """, [('0.5ml', 'Salina'), ('1.0ml', 'Cafeína'), ('0.2ml', 'Alcohol')])
            conn.commit()
            print("Datos insertados en la tabla Dosis")
    except Exception as e:
        print(f"Error al insertar en Dosis: {e}")
    finally:
        cursor.close()
        conn.close()


def insert_video(video_name, id_raton=1, nro_muestra=101, id_tipo_prueba=1, id_dosis=1):
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

        # Insertar el nuevo registro
        cursor.execute("""
            INSERT INTO Video (idVideo, idRaton, nroMuestra, idTipoPrueba, idDosis)
            VALUES (?, ?, ?, ?, ?)
        """, (video_name, id_raton, nro_muestra, id_tipo_prueba, id_dosis))
        conn.commit()
        print(f"Video '{video_name}' insertado correctamente.")
    except Exception as e:
        print(f"Error al insertar el video: {e}")
    finally:
        cursor.close()
        conn.close()


def insert_results_to_db(distances, area_central_data, times_data, trajectory_data, video_name):
    """
    Inserta los resultados del procesamiento en las tablas Trayectoria y TiempoCuriosidad.

    Args:
        distances (dict): Diccionario con las distancias recorridas por los keypoints.
        area_central_data (dict): Diccionario con datos del área central (distancia, entradas, salidas, tiempo).
        times_data (dict): Diccionario con los tiempos en zonas de interés.
        trajectory_data (list): Lista de diccionarios con los mapas de trayectoria como imágenes en bytes.
        video_name (str): Nombre del video procesado.
    """
    # Verificar si las tablas relacionadas están llenas
    insert_raza()
    insert_raton()
    insert_dosis()
    insert_video(video_name)

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
        for traj in trajectory_data:  # Iterar sobre los elementos de la lista
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
                INSERT INTO Trayectoria (idVideo, mapaTrayectoria, distanciaRecorrida, descripcion, area_central, nro_entrada_area, nro_salida_area)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                id_video,
                trajectory_image,       # Mapa de trayectoria en bytes
                distance,               # Distancia recorrida total
                keypoint,               # Nombre del keypoint
                area_central_distance,  # Distancia dentro del área central
                entries,                # Número de entradas al área central
                exits                   # Número de salidas del área central
            ))
        print("Datos insertados correctamente en la tabla Trayectoria")

        # Insertar en la tabla TiempoCuriosidad
        for objeto_interes, tiempo_curiosidad in times_data.items():
            cursor.execute("""
                INSERT INTO TiempoCuriosidad (idVideo, objetoInteres, tiempoCuriosidad)
                VALUES (?, ?, ?)
            """, (
                id_video,
                objeto_interes,  # Nombre del objeto de interés
                tiempo_curiosidad  # Tiempo de curiosidad en segundos
            ))
        print(f"Datos insertados correctamente en la tabla TiempoCuriosidad para el video '{video_name}'.")

        conn.commit()
        print(f"Resultados insertados correctamente en la base de datos para el video '{video_name}'.")
    except Exception as e:
        print(f"Error al insertar datos en la base de datos: {e}")
    finally:
        cursor.close()
        conn.close()
