const socket = io('http://127.0.0.1:5000');

// Verifica cuando el cliente se conecta al servidor de WebSockets
socket.on('connect', () => {
  console.log('Conectado al servidor de WebSockets');
});

// Verifica cuando el cliente se desconecta del servidor de WebSockets
socket.on('disconnect', () => {
  console.log('Desconectado del servidor de WebSockets');
});

socket.on('progress_update', (progressData) => {
  console.log('Progreso recibido:', progressData);
  if (progressData && typeof progressData.progress !== 'undefined') {
    updateProgressBar(progressData.progress);
  } else {
    console.error('Datos de progreso no válidos recibidos:', progressData);
  }
});

// Emitir un evento de prueba
socket.emit('test_event', { test: 'data' });

// Escuchar el evento de respuesta del servidor
socket.on('test_response', (data) => {
    console.log('Respuesta del servidor:', data);
});

// Función para alternar los campos adicionales de dosis
function toggleDoseFields() {
  const doseFields = document.querySelector('.dose-fields');
  const doseSelect = document.getElementById('dose').value;

  if (doseSelect === 'Dosis Aplicada') {
      doseFields.classList.remove('hidden');
  } else {
      doseFields.classList.add('hidden');
  }
}

// ################################## Modales ####################################

// Abre el modal con un título personalizado
function openModal(action) {
    currentAction = action;
    const modal = document.getElementById('modal-container');
    const modalTitle = document.getElementById('modal-title');
    const modalInput = document.getElementById('modal-input');

    // Personaliza el título y el placeholder del modal
    if (action === "breed") {
        modalTitle.innerText = "Nueva Raza";
        modalInput.placeholder = "Ingrese el nombre de la raza";
    } else if (action === "dose") {
        modalTitle.innerText = "Nueva Dosis";
        modalInput.placeholder = "Ingrese el nombre de la dosis";
    }

    modalInput.value = ""; // Limpia el campo de entrada
    modal.classList.remove('hidden'); // Muestra el modal
}

// Cierra el modal
function closeModal() {
    const modal = document.getElementById('modal-container');
    modal.classList.add('hidden'); // Oculta el modal
}

// Modal de Éxito
function showSuccessModal(message) {
  const modal = document.getElementById('success-modal');
  const messageElement = document.getElementById('success-message');
  messageElement.textContent = message; // Establece el mensaje
  modal.classList.remove('hidden'); // Muestra el modal
}

function closeSuccessModal() {
  const modal = document.getElementById('success-modal');
  modal.classList.add('hidden'); // Oculta el modal
}

// Modal de Alerta
function showAlertModal(message) {
  const modal = document.getElementById('alert-modal');
  const messageElement = document.getElementById('alert-message');
  messageElement.textContent = message; // Establece el mensaje
  modal.classList.remove('hidden'); // Muestra el modal
}

function closeAlertModal() {
  const modal = document.getElementById('alert-modal');
  modal.classList.add('hidden'); // Oculta el modal
}

// Modal de Error
function showErrorModal(message) {
  const modal = document.getElementById('error-modal');
  const messageElement = document.getElementById('error-message');
  messageElement.textContent = message; // Establece el mensaje
  modal.classList.remove('hidden'); // Muestra el modal
}

function closeErrorModal() {
  const modal = document.getElementById('error-modal');
  modal.classList.add('hidden'); // Oculta el modal
}

let currentAction = ""; // Variable para identificar si es raza o dosis
// Guarda el valor ingresado en el modal
function confirmModal() {
  const modalInput = document.getElementById('modal-input');
  const value = modalInput.value.trim();

  if (value) {
      if (currentAction === "breed") {
          const breedSelect = document.getElementById('breed');
          const newOption = document.createElement('option');
          newOption.value = value;
          newOption.textContent = value;
          breedSelect.appendChild(newOption);

          // Mostrar el modal de Éxito
          showSuccessModal(`Raza "${value}" agregada con éxito.`);
      } else if (currentAction === "dose") {
          const doseList = document.getElementById('dose-list');
          const newOption = document.createElement('option');
          newOption.value = value;
          newOption.textContent = value;
          doseList.appendChild(newOption);

          // Mostrar el modal de Éxito
          showSuccessModal(`Dosis "${value}" agregada con éxito.`);
      }

      closeModal(); // Cierra el modal de entrada
  } else {
      // Mostrar el modal de Alerta
      showAlertModal("Por favor, ingrese un nombre válido.");
  }
}

// Abre el modal de confirmación con un mensaje personalizado
function showConfirmationModal(message) {
  const confirmationModal = document.getElementById('confirmation-modal');
  const confirmationMessage = document.getElementById('confirmation-message');

  confirmationMessage.textContent = message; // Establece el mensaje
  confirmationModal.classList.remove('hidden'); // Muestra el modal
}

// Cierra el modal de confirmación
function closeConfirmationModal() {
  const confirmationModal = document.getElementById('confirmation-modal');
  confirmationModal.classList.add('hidden'); // Oculta el modal
}

function displayFileDetails() {
  const fileInput = document.getElementById('file-input');
  const fileUploadBox = document.getElementById('file-upload-box');

  if (fileInput.files.length > 0) {
      const selectedFile = fileInput.files[0];
      const fileType = selectedFile.type;

      // Limpia el contenido actual del área de carga
      fileUploadBox.innerHTML = '';

      if (fileType.startsWith('video/')) {    
        
        // Cargar un frame del video
        const videoElement = document.createElement('video');
        videoElement.src = URL.createObjectURL(selectedFile);
        videoElement.preload = 'metadata';
        videoElement.muted = true;

        const canvasElement = document.createElement('canvas');
        const context = canvasElement.getContext('2d');

        videoElement.addEventListener('loadeddata', () => {
            // Establecer dimensiones del canvas basadas en el video
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;

            // Capturar el primer frame
            context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

            // Crear la imagen a partir del canvas
            const imgElement = document.createElement('img');
            imgElement.src = canvasElement.toDataURL('image/png');
            imgElement.alt = "Miniatura del video";
            imgElement.id = "file-input";
            imgElement.type = "file";
            imgElement.style.maxWidth = '100%';
            imgElement.style.maxHeight = '100%';
            imgElement.style.borderRadius = '10px';

            // Permitir hacer clic en la miniatura para cambiar el video
            imgElement.style.cursor = 'pointer';
            imgElement.onclick = () => fileInput.click(); // Reabrir el selector de archivos

            // Añadir la miniatura al área de carga
            fileUploadBox.appendChild(imgElement);

            // Pausar el video después de capturar el frame
            videoElement.pause();
        });

        // Reproduce el video en segundo plano para asegurar la carga del primer frame
        videoElement.play();
      } else {
          showErrorModal("El archivo seleccionado no es un video. \nPor favor seleccionar un archivo con formato de video.");
      }
  }
}

// Función para redirigir a otra página (Botón "Atrás")
function goBack() {
  window.history.back();
}

//###################################### BARRA DE PROGRESO #######################################
let processingInterval;
let estimatedTotalTime = 10; // Tiempo estimado inicial en segundos
let startTime; // Tiempo de inicio del procesamiento

function startProcessing() {
  const progressModal = document.getElementById('progress-modal');
  const fileInput = document.getElementById('file-input');

  if (!fileInput.files || fileInput.files.length === 0) {
      showErrorModal("Por favor, carga un video antes de iniciar el procesamiento.");
      return;
  }

  progressModal.classList.remove('hidden');

  const videoFile = fileInput.files[0];
  const breed = document.getElementById('breed').value;
  const gender = document.getElementById('gender').value;
  const dose = document.getElementById('dose').value;
  const doseList = document.getElementById('dose-list').value;
  const doseAmount = document.getElementById('dose-amount').value;

  let formData = new FormData();
  formData.append('videoFile', videoFile);
  formData.append('breed', breed);
  formData.append('gender', gender);
  formData.append('dose', dose);
  formData.append('doseList', doseList);
  formData.append('doseAmount', doseAmount);


  // Luego, realiza la solicitud para cargar el video
  fetch('http://127.0.0.1:5000/upload_video', {
    method: 'POST',
    body: formData,
  })
  .then(response => response.json())
  .then(data => {
    if (data.status === 'success') {
        console.log('El video se está procesando en el backend.');
    } else {
        console.error('Error al procesar el video:', data.message);
    }
  })
  .catch(error => {
    console.error('Error en la solicitud:', error);
  });
}

function updateProgressBar(progress) {
  const progressBar = document.getElementById('progress-bar');
  const progressPercentageText = document.getElementById('progress-percentage-text');
  const progressPercentageBar = document.getElementById('progress-percentage-bar');

  // Actualiza la barra de progreso y el texto
  progressBar.style.width = `${progress}%`;
  progressPercentageText.innerText = `${Math.floor(progress)}%`;
  progressPercentageBar.innerText = `${Math.floor(progress)}%`;

  // Si se alcanza el 100% de progreso, se finaliza el procesamiento
  if (progress >= 100) {
      clearInterval(processingInterval);
      finishProcessing();
  }
}

// function updateProgressBar() {
//     const currentTime = Date.now();
//     const elapsedTime = (currentTime - startTime) / 1000; // Tiempo transcurrido en segundos
//     const remainingTime = Math.max(0, estimatedTotalTime - elapsedTime); // Tiempo restante en segundos

//     // Calcula el progreso como porcentaje dinámico
//     const progress = Math.min(100, (elapsedTime / estimatedTotalTime) * 100);
//     const progressBar = document.getElementById('progress-bar');
//     const progressPercentageText = document.getElementById('progress-percentage-text');
//     const progressPercentageBar = document.getElementById('progress-percentage-bar');
//     const estimatedTime = document.getElementById('estimated-time');

//     // Actualiza la barra de progreso y el texto
//     progressBar.style.width = `${progress}%`;
//     progressPercentageText.innerText = `${Math.floor(progress)}%`;
//     progressPercentageBar.innerText = `${Math.floor(progress)}%`;

//     // Convierte el tiempo restante a hh:mm:ss
//     const hours = Math.floor(remainingTime / 3600);
//     const minutes = Math.floor((remainingTime % 3600) / 60);
//     const seconds = Math.floor(remainingTime % 60);
//     estimatedTime.innerText = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

//     // Si se alcanza el 100% de progreso, se finaliza el procesamiento
//     if (progress >= 100) {
//         clearInterval(processingInterval);
//         finishProcessing();
//     }
// }

function finishProcessing() {
    const progressModal = document.getElementById('progress-modal');
    progressModal.classList.add('hidden'); // Ocultar el modal de progreso

    const completionModal = document.getElementById('completionModal');
    completionModal.classList.remove('hidden'); // Mostrar el modal de finalización
}

function cancelProcessing() {
    const confirmCancel = confirm("¿Estás seguro de cancelar el procesamiento?");
    if (confirmCancel) {
        clearInterval(processingInterval);
        const progressModal = document.getElementById('progress-modal');
        progressModal.classList.add('hidden'); // Oculta el modal de progreso

        const progressBar = document.getElementById('progress-bar');
        const progressPercentageText = document.getElementById('progress-percentage-text');
        const progressPercentageBar = document.getElementById('progress-percentage-bar');
        const estimatedTime = document.getElementById('estimated-time');

        // Reinicia los valores
        progressBar.style.width = '0%';
        progressPercentageText.innerText = '0%';
        progressPercentageBar.innerText = '0%';
        estimatedTime.innerText = '00:00:00';
    }
}

// Función para actualizar dinámicamente el tiempo estimado de finalización
function updateEstimatedTime(newEstimatedTime) {
    estimatedTotalTime = newEstimatedTime; // Actualiza el tiempo total estimado dinámicamente
}

// Redirige a la página de resultados después de procesar
function redirectToResults() {
  window.location.href = 'results.html';
}

// Recarga la página para subir otro video
function uploadAnotherVideo() {
  window.location.reload();
}
