@echo off
setlocal

echo [INFO] Buscando "activate.bat" en:
echo         C:\Anaconda\Scripts\activate.bat
echo         D:\Anaconda\Scripts\activate.bat

REM Variable para guardar la ruta encontrada
set "FOUND_ACTIVATE_BAT="

REM 1) Revisar en C:
IF EXIST "C:\Anaconda\Scripts\activate.bat" (
    set "FOUND_ACTIVATE_BAT=C:\Anaconda\Scripts\activate.bat"
    goto :FOUND_IT
)

REM 2) Revisar en D:
IF EXIST "D:\Anaconda\Scripts\activate.bat" (
    set "FOUND_ACTIVATE_BAT=D:\Anaconda\Scripts\activate.bat"
    goto :FOUND_IT
)

:FOUND_IT

REM 3) ¿Se encontró?
IF NOT DEFINED FOUND_ACTIVATE_BAT (
    echo [ERROR] No se encontró "activate.bat" en C:\Anaconda\Scripts ni en D:\Anaconda\Scripts
    pause
    exit /B 1
)

echo [INFO] Se encontró "activate.bat" en: %FOUND_ACTIVATE_BAT%

REM 4) Activar TesisHRNet en ESTA consola
CALL "%FOUND_ACTIVATE_BAT%" TesisHRNet

REM 5) Abrir dos ventanas con el entorno activado en cada una
start cmd.exe /K "CALL "%FOUND_ACTIVATE_BAT%" TesisHRNet && cd /d %~dp0STUART\scripts && python app.py"
start cmd.exe /K "CALL "%FOUND_ACTIVATE_BAT%" TesisHRNet && cd /d %~dp0STUART-Frontend\src && python -m http.server 8000"

echo [INFO] Se han abierto dos ventanas:
echo        - Backend (app.py)
echo        - Frontend (puerto 8000)

pause