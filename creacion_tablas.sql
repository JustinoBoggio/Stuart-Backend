CREATE TABLE Usuario (
    mail VARCHAR(100) PRIMARY KEY,
    contraseña VARCHAR(255) NOT NULL
);

CREATE TABLE Raza (
    idRaza INT IDENTITY(1,1) PRIMARY KEY,
    nombreRaza VARCHAR(100)
);

CREATE TABLE Raton (
    idRaton INT IDENTITY(1,1) PRIMARY KEY,
    sexo VARCHAR(45),
    idRaza INT,
    FOREIGN KEY (idRaza) REFERENCES Raza(idRaza)
);

CREATE TABLE Dosis (
    idDosis INT IDENTITY(1,1) PRIMARY KEY,
    descripcion VARCHAR(150)
);

CREATE TABLE Video (
    idVideo VARCHAR(100) PRIMARY KEY,
    idRaton INT,
    idDosis INT,
	cantidad VARCHAR(45) NULL,
    mail_usuario VARCHAR(100),

    FOREIGN KEY (idRaton) REFERENCES Raton(idRaton),
    FOREIGN KEY (idDosis) REFERENCES Dosis(idDosis),
    FOREIGN KEY (mail_usuario) REFERENCES Usuario(mail)
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
    descripcion VARCHAR(100),
    distanciaRecorrida_AC FLOAT,
    nro_entrada_AC INT,
    nro_salida_AC INT,
	tiempo_dentro_AC FLOAT,
    CONSTRAINT FK_Trayectoria_Video
        FOREIGN KEY (idVideo) REFERENCES Video(idVideo)
        ON DELETE CASCADE
);