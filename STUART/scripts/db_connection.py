# db_connection.py
import pyodbc

def get_db_connection():
    #Luz
    # driver = SQL Server
    #server = 'AR-IT13485'  # por ejemplo, 'localhost' o '192.168.1.1'
    #database = 'STUART1'  # el nombre de tu base de datos
    #username = 'luz_audi'  # tu usuario de SQL Server
    #password = 'recibida2024'   # tu contraseña de SQL Server

    #Justino
    # driver = SQL Server
    # server = 'DESKTOP-9M0HA0J\SQLSERVERJUSTINO'
    # database = 'STUART'
    # username = 'sa'
    # password = 'root'

    #Leonel
    driver = '{ODBC Driver 17 for SQL Server}'
    server = 'DESKTOP-LBNEVAD'
    database = 'STUART'
    username = 'sa'
    password = 'Lani01020608'

    # Establecer la conexión
    try:
        conn = pyodbc.connect(f'Driver={driver};'
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