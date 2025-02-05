CREATE TABLE Usuario (
    mail VARCHAR(45) PRIMARY KEY,
    contraseña VARCHAR(255) NOT NULL
);

CREATE TABLE Raza (
    idRaza INT IDENTITY(1,1) PRIMARY KEY,
    nombreRaza VARCHAR(45)
);

CREATE TABLE Raton (
    idRaton INT IDENTITY(1,1) PRIMARY KEY,
    sexo VARCHAR(45),
    idRaza INT,
    FOREIGN KEY (idRaza) REFERENCES Raza(idRaza)
);

CREATE TABLE Dosis (
    idDosis INT IDENTITY(1,1) PRIMARY KEY,
    cantidad VARCHAR(45),
    sustancia VARCHAR(45)
);

CREATE TABLE Video (
    idVideo VARCHAR(100) PRIMARY KEY,
    idRaton INT,
    nroMuestra INT,
    idTipoPrueba INT,
    idDosis INT,
    mail_usuario VARCHAR(45),
    FOREIGN KEY (idRaton) REFERENCES Raton(idRaton),
    FOREIGN KEY (idDosis) REFERENCES Dosis(idDosis),
    FOREIGN KEY (mail_usuario) REFERENCES Usuario(mail_usuario)
);

CREATE TABLE TiempoCuriosidad (
    idTiempoCuriosidad INT IDENTITY(1,1) PRIMARY KEY,
    idVideo VARCHAR(100),
    objetoInteres VARCHAR(45),
    tiempoCuriosidad FLOAT,
    CONSTRAINT FK_TiempoCuriosidad_Video
        FOREIGN KEY (idVideo) REFERENCES Video(idVideo)
        ON DELETE CASCADE
);

CREATE TABLE Trayectoria (
    idTrayectoria INT IDENTITY(1,1) PRIMARY KEY,
    idVideo VARCHAR(100),
    mapaTrayectoria VARBINARY(MAX),
    distanciaRecorrida FLOAT,
    descripcion VARCHAR(45),
    distanciaRecorrida_AC FLOAT,
    nro_entrada_AC INT,
    nro_salida_AC INT,
	tiempo_dentro_AC FLOAT,
    CONSTRAINT FK_Trayectoria_Video
        FOREIGN KEY (idVideo) REFERENCES Video(idVideo)
        ON DELETE CASCADE
);
