# Asegúrate de que el script se detenga en caso de error
$ErrorActionPreference = "Stop"

# Mostrar mensaje de inicio
Write-Host "Iniciando script..."

# Verificar la versión de conda
try {
    conda --version
    Write-Host "Conda está disponible."
} catch {
    Write-Error "Conda no está disponible. Asegúrate de que Anaconda o Miniconda esté instalado y que 'conda' esté en el PATH."
    exit 1
}

# Activar el entorno de conda
try {
    & conda activate TesisHrnet
    Write-Host "Entorno 'TesisHrnet' activado."
} catch {
    Write-Error "No se pudo activar el entorno 'TesisHrnet'. Asegúrate de que el entorno exista."
    exit 1
}

# Navegar al directorio del backend y ejecutarlo
try {
    Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "cd \"$(Join-Path $PSScriptRoot 'STUART\scripts'); python app.py\""
    Write-Host "Backend iniciado."
} catch {
    Write-Error "No se pudo iniciar el backend."
}

# Navegar al directorio del frontend y ejecutarlo
try {
    Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "cd \"$(Join-Path $PSScriptRoot 'STUART-Frontend\src'); python -m http.server 8000\""
    Write-Host "Frontend iniciado."
} catch {
    Write-Error "No se pudo iniciar el frontend."
}

Write-Host "Script finalizado."