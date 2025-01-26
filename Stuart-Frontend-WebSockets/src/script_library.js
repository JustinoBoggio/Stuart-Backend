function getCurrentDate() {
    const today = new Date();
    const day = today.getDate();
    const month = today.getMonth() + 1; // Los meses comienzan desde 0
    const year = today.getFullYear();

    // Actualizar el contenido del elemento con el ID "current-date" con la fecha actual
    document.getElementById('current-date').textContent = `${day}/${month}/${year}`;
  }


// Simula datos de la base de datos
const videosData = [
    { name: 'Reconocimiento_1_F', sex: 'Masculino' },
    { name: 'Test_1_F', sex: 'Masculino' },
    { name: 'Reconocimiento_1_A', sex: 'Femenino' },
    { name: 'Test_1_A', sex: 'Femenino' },
    // Agrega más datos según sea necesario
];

// Función para cargar la lista de videos
function loadVideoList() {
    const videoListContainer = document.getElementById('videoList');

    // Limpia la lista antes de agregar nuevos elementos
    videoListContainer.innerHTML = '';

    if (videosData.length === 0) {
        // No hay resultados, mostrar mensaje
        const noResultsTitle = document.createElement('h2');
        noResultsTitle.textContent = 'No hay resultados disponibles';
        videoListContainer.appendChild(noResultsTitle);
    } else {
        // Itera sobre los datos y crea las filas de videos
        videosData.forEach(video => {
            const videoItem = document.createElement('div');
            videoItem.classList.add('video-item');

            // Icono de video
            const videoIcon = document.createElement('div');
            videoIcon.classList.add('video-icon');
            videoIcon.innerHTML = '<i class="fas fa-video"></i>'; // Puedes cambiar el icono

            // Nombre del video
            const videoName = document.createElement('div');
            videoName.textContent = video.name;

            // Sexo
            const videoSex = document.createElement('div');
            videoSex.textContent = video.sex;

            // Botón "Ver Resultados"
            const viewResultsBtn = document.createElement('button');
            viewResultsBtn.textContent = 'Ver Resultados';
            viewResultsBtn.setAttribute('href', 'index.html?source=library');
            viewResultsBtn.addEventListener('click', () => viewResults(video.name));

            // Botón "Eliminar"
            const deleteBtn = document.createElement('button');
            deleteBtn.innerHTML = '<i class="fas fa-trash"></i>';
            deleteBtn.classList.add('delete-button');
            deleteBtn.addEventListener('click', () => deleteVideo(video.name));

            // Agrega los elementos a la fila de video
            videoItem.appendChild(videoIcon);
            videoItem.appendChild(videoName);
            videoItem.appendChild(videoSex);
            videoItem.appendChild(viewResultsBtn);
            videoItem.appendChild(deleteBtn);

            // Agrega la fila de video al contenedor
            videoListContainer.appendChild(videoItem);
        });
    }
}


// Función para simular ver resultados
function viewResults(videoName) {
    //alert(`Ver resultados para el video: ${videoName}`);
    //logica para llevar el id del video y ver esos datos
    window.location.href = 'results.html';

}

// Función para simular eliminar video con modal
function deleteVideo(videoName) {
    const modalText = `¿Estás seguro de eliminar el video: ${videoName}?`;

    showModal(modalText, () => {
        // Elimina el video de la lista (simulado)
        videosData.splice(videosData.findIndex(video => video.name === videoName), 1);
        // Vuelve a cargar la lista actualizada
        loadVideoList();
    });
}

function showModal(text, onConfirm) {
    const modalContainer = document.createElement('div');
    modalContainer.classList.add('modal-container');

    const modalContent = document.createElement('div');
    modalContent.classList.add('modal-content');
    modalContent.innerHTML = `
        <p>${text}</p>
        <button id="confirmBtn" class="confirmBtn">Confirmar</button>
        <button id="cancelBtn" class="cancelBtn">Cancelar</button>
    `;

    modalContainer.appendChild(modalContent);
    document.body.appendChild(modalContainer);

    // Agrega eventos a los botones
    const confirmBtn = document.getElementById('confirmBtn');
    const cancelBtn = document.getElementById('cancelBtn');

    confirmBtn.addEventListener('click', () => {
        if (onConfirm) {
            onConfirm();
        }
        closeModal();
    });

    cancelBtn.addEventListener('click', () => {
        closeModal();
    });
}

// Función para cerrar el modal
function closeModal() {
    const modalContainer = document.querySelector('.modal-container');
    if (modalContainer) {
        modalContainer.remove();
    }
}


// Función para retroceder (puede ser personalizada según tu navegación)
function goBack() {
    window.history.back();
}

// Llama a la función para cargar la lista de videos al cargar la página
window.onload = function () {
    getCurrentDate()
    loadVideoList();
  };
