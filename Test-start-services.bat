@echo off
setlocal

echo "Iniciando script..."
cmd /k "conda --version"

echo "Verificando Conda..."
conda --version >nul 2>&1
if errorlevel 1 (
    echo "Conda no está disponible. Asegúrate de que Anaconda o Miniconda esté instalado y que 'conda' esté en el PATH."
    pause
    exit /b 1
)

echo "Inicializando Conda..."
conda init >nul 2>&1

if errorlevel 0 (
    echo "Conda ha sido inicializado. Por favor, cierre esta ventana y vuelva a ejecutar el script."
    pause
    exit /b 1
)

echo "Activando el entorno 'TesisHrnet'..."
call conda activate TesisHrnet

if not "%CONDA_DEFAULT_ENV%"=="TesisHrnet" (
    echo "No se pudo activar el entorno 'TesisHrnet'. Asegúrate de que el entorno exista."
    pause
    exit /b 1
)

echo "Entorno activado correctamente."

echo "Iniciando el backend..."
start cmd.exe /k "cd /d %~dp0STUART\scripts && python app.py"

echo "Iniciando el frontend..."
start cmd.exe /k "cd /d %~dp0STUART-Frontend\src && python -m http.server 8000"

echo "Script finalizado."
pause
endlocal