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

    // Actualiza el tiempo restante si está disponible
    if (typeof progressData.remaining_time !== 'undefined') {
      updateRemainingTime(progressData.remaining_time);
    }
  } else {
    console.error('Datos de progreso no válidos recibidos:', progressData);
  }
});

document.addEventListener('DOMContentLoaded', function() {
  loadBreeds();
  loadDoses();
});

window.addEventListener('beforeunload', function (e) {
  // Solo marca la cancelación si el procesamiento había comenzado
  if (localStorage.getItem('processingStarted') === 'true') {
    localStorage.setItem('analysisCancelled', 'true');
    fetch('http://127.0.0.1:5000/cancel_analysis', { method: 'POST' });
  }
});

// Al cargar la página nuevamente
window.addEventListener('load', function () {
  // Verifica si el procesamiento había comenzado y se canceló
  if (localStorage.getItem('processingStarted') === 'true' && localStorage.getItem('analysisCancelled') === 'true') {
    showAlertModal('El análisis anterior fue cancelado.');

    // Limpia los indicadores
    localStorage.removeItem('analysisCancelled');
    localStorage.removeItem('processingStarted');
  }
});

//Funciones para Dosis y Raza
function loadBreeds() {
  fetch('http://127.0.0.1:5000/api/breeds')
      .then(response => {
          if (!response.ok) {
              throw new Error('Network response was not ok');
          }
          return response.json();
      })
      .then(data => {
          const breedSelect = document.getElementById('breed');
          breedSelect.innerHTML = ''; // Clear existing options

          // Agrega la opción predeterminada
          const defaultOption = document.createElement('option');
          defaultOption.value = ""; // Valor vacío para que no represente una opción válida
          defaultOption.textContent = "Seleccione una raza";
          defaultOption.disabled = true; // Desactivar la opción para que no pueda seleccionarse
          defaultOption.selected = true; // Seleccionar como opción predeterminada
          breedSelect.appendChild(defaultOption);

          data.forEach(breed => {
              const option = document.createElement('option');
              option.value = breed.idRaza;
              option.textContent = breed.nombreRaza;
              breedSelect.appendChild(option);
          });
      })
      .catch(error => console.error('Error al cargar razas:', error));
}

function loadDoses() {
  fetch('http://127.0.0.1:5000/api/doses')
      .then(response => {
          if (!response.ok) {
              throw new Error('Network response was not ok');
          }
          return response.json();
      })
      .then(data => {
          const doseSelect = document.getElementById('dose-list');
          doseSelect.innerHTML = ''; // Clear existing options

          // Agrega la opción predeterminada
          const defaultOption = document.createElement('option');
          defaultOption.value = ""; // Valor vacío para que no represente una opción válida
          defaultOption.textContent = "Seleccione una dosis";
          defaultOption.disabled = true; // Desactivar la opción para que no pueda seleccionarse
          defaultOption.selected = true; // Seleccionar como opción predeterminada
          doseSelect.appendChild(defaultOption);

          data.forEach(dose => {
              const option = document.createElement('option');
              option.value = dose.idDosis;
              option.textContent = dose.descripcion;
              doseSelect.appendChild(option);
          });
      })
      .catch(error => console.error('Error al cargar dosis:', error));
}

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


function confirmModal() {
  const modalInput = document.getElementById('modal-input');
  const value = modalInput.value.trim();

  if (value) {
      let url = "";
      const data = { name: value };

      if (currentAction === "breed") {
          url = 'http://127.0.0.1:5000/api/add_breed';
      } else if (currentAction === "dose") {
          url = 'http://127.0.0.1:5000/api/add_dose';
      }

      fetch(url, {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify(data)
      })
      .then(response => {
          if (!response.ok) {
              throw new Error('Error al agregar elemento');
          }
          return response.json();
      })
      .then(data => {
          console.log(data.message);
          if (currentAction === "breed") {
              const breedSelect = document.getElementById('breed');
              const newOption = document.createElement('option');
              newOption.value = value;
              newOption.textContent = value;
              breedSelect.appendChild(newOption);
              showSuccessModal(`Raza "${value}" agregada con éxito.`);
          } else if (currentAction === "dose") {
              const doseList = document.getElementById('dose-list');
              const newOption = document.createElement('option');
              newOption.value = value;
              newOption.textContent = value;
              doseList.appendChild(newOption);
              showSuccessModal(`Dosis "${value}" agregada con éxito.`);
          }
          closeModal(); // Cierra el modal de entrada
      })
      .catch(error => {
          console.error('Error:', error);
          showAlertModal('Hubo un problema al agregar el elemento.');
      });
  } else {
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

function cancelProcessing() {
  const confirmCancel = confirm("¿Estás seguro de cancelar el procesamiento?");
  if (confirmCancel) {
    // Detener cualquier temporizador o intervalo
    if (typeof processingInterval !== 'undefined') {
      clearInterval(processingInterval);
      console.log('Intervalo detenido');
    }

    // Ocultar el modal de progreso
    const progressModal = document.getElementById('progress-modal');
    if (progressModal) {
      progressModal.classList.add('hidden');
    }

    // Reiniciar los valores de progreso en la interfaz de usuario
    const progressBar = document.getElementById('progress-bar');
    const progressPercentageText = document.getElementById('progress-percentage-text');
    const progressPercentageBar = document.getElementById('progress-percentage-bar');
    const estimatedTime = document.getElementById('estimated-time');

    if (progressBar) {
      progressBar.style.width = '0%';
      // Forzar re-renderizado
      progressBar.offsetHeight; // Leer una propiedad del DOM
    }
    if (progressPercentageText) {
      progressPercentageText.innerText = '0%';
    }
    if (progressPercentageBar) {
      progressPercentageBar.innerText = '0%';
    }
    if (estimatedTime) {
      estimatedTime.innerText = '00:00:00';
    }
    localStorage.removeItem('analysisCancelled');
    localStorage.removeItem('processingStarted');

    // Enviar solicitud al backend para cancelar el análisis
    fetch('http://127.0.0.1:5000/cancel_analysis', { method: 'POST' })
      .then(response => {
        if (!response.ok) {
          throw new Error('Error al cancelar el análisis');
        }
        return response.json();
      })
      .then(data => {
        console.log(data.status);
        showAlertModal('El análisis fue cancelado.'); // Muestra un mensaje de alerta
      })
      .catch(error => {
        console.error('Error:', error);
        showAlertModal('Hubo un problema al cancelar el análisis.');
      });
  }
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
let lastProcessedVideo = null;

function startProcessing() {
  localStorage.setItem('processingStarted', 'true');
  const progressModal = document.getElementById('progress-modal');
  const fileInput = document.getElementById('file-input');

  if (!fileInput.files || fileInput.files.length === 0) {
      showErrorModal("Por favor, carga un video antes de iniciar el procesamiento.");
      return;
  }

  progressModal.classList.remove('hidden');

  const videoFile = fileInput.files[0];
  lastProcessedVideo = videoFile.name.replace(/\.[^/.]+$/, ""); 
  
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

function updateRemainingTime(remainingTime) {
  const estimatedTime = document.getElementById('estimated-time');
  if (estimatedTime) {
    // Actualiza el texto del elemento HTML con el tiempo restante recibido
    estimatedTime.innerText = remainingTime;
  } else {
    console.error('Elemento con ID "estimated-time" no encontrado.');
  }
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

function finishProcessing() {
    const progressModal = document.getElementById('progress-modal');
    progressModal.classList.add('hidden'); // Ocultar el modal de progreso

    const completionModal = document.getElementById('completionModal');
    completionModal.classList.remove('hidden'); // Mostrar el modal de finalización
}


// Función para actualizar dinámicamente el tiempo estimado de finalización
function updateEstimatedTime(newEstimatedTime) {
    estimatedTotalTime = newEstimatedTime; // Actualiza el tiempo total estimado dinámicamente
}

// Redirige a la página de resultados después de procesar
function redirectToResults() {
  if (!lastProcessedVideo) {
      console.error("Error: No hay un video procesado.");
      showErrorModal("No se ha encontrado un video procesado. Asegúrate de procesar un video primero.");
      return;
  }

  window.location.href = `results.html?video=${encodeURIComponent(lastProcessedVideo)}`;
}

// Recarga la página para subir otro video
function uploadAnotherVideo() {
  localStorage.removeItem('analysisCancelled');
  localStorage.removeItem('processingStarted');
  window.location.reload();
}
