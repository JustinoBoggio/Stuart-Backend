// Establecer la conexión con el servidor de Socket.IO
const socket = io('http://127.0.0.1:5000');

// Función para mostrar el modal
function showModal() {
  document.getElementById('progressModal').style.display = 'block';
}

// Función para cerrar el modal
function closeModal() {
  document.getElementById('progressModal').style.display = 'none';
}

// Función para actualizar el progreso en el modal
function updateProgress(percentage) {
  const progressBarFill = document.getElementById('progress-bar-fill');
  const progressText = document.getElementById('progress-text');
  progressBarFill.style.width = percentage + '%';
  progressText.textContent = percentage + '%';
}

// Función que maneja el envío de los datos del formulario
  function sendVideoData() {
    console.log("Envío de datos de video iniciado");
    
    const videoInput = document.getElementById('file-input');
    if (!videoInput || !videoInput.files || !videoInput.files[0]) {
        alert("Por favor, selecciona un archivo de video.");
        return;
    }
    const videoFile = videoInput.files[0];

    const breed = document.getElementById('breed').value;
    const gender = Array.from(document.querySelectorAll('input[name="gender"]:checked')).map(input => input.value);
    const proof = Array.from(document.querySelectorAll('input[name="proof"]:checked')).map(input => input.value);
    const dosisCantidad = document.getElementById('dosisCantidad').value;
    const dosisNombre = document.getElementById('dosisNombre').value;

    let formData = new FormData();
    formData.append('videoFile', videoFile);
    formData.append('breed', breed);
    formData.append('gender', gender.join(','));
    formData.append('proof', proof.join(','));
    formData.append('dosisCantidad', dosisCantidad);
    formData.append('dosisNombre', dosisNombre);

    formData.forEach((value, key) => {
        console.log(key, value);
    });

    console.log("Enviando solicitud POST al backend");
    fetch('http://127.0.0.1:5000/upload_video', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        console.log('Datos del servidor:', data);
        if (data.status === 'success') {
            showModal();

            // Escuchar los eventos de progreso
            socket.on('progress_update', (data) => {
                updateProgress(data.progress);

                if (data.progress >= 100) {
                    alert('Video analysis complete!');
                    closeModal();
                }
            });
        }
    })
    .catch(error => {
        console.error('Error en la solicitud POST:', error);
    });
  }

function redirectToAnotherPage() {
  window.location.href = 'index.html';
}