# db_connection.py
import pyodbc

def get_db_connection():
    # Configura los parámetros de conexión
    server = 'DESKTOP-9M0HA0J\SQLSERVERJUSTINO'  # por ejemplo, 'localhost' o '192.168.1.1'
    database = 'STUART'  # el nombre de tu base de datos
    username = 'sa'  # tu usuario de SQL Server
    password = 'root'  # tu contraseña de SQL Server

    # Establecer la conexión
    try:
        conn = pyodbc.connect('Driver={SQL Server};'
                              f'Server={server};'
                              f'Database={database};'
                              f'UID={username};'
                              f'PWD={password};')
        print("Conexión exitosa a SQL Server")
        return conn
    except Exception as e:
        print(f"Error al conectar a SQL Server: {e}")
        return None
    
#get_db_connection()