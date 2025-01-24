// Simula datos de la base de datos
const videosData = [
    { name: 'Test_1_F_Recortado', sex: 'Masculino', race: 'Raza 1', dose: 'Dosis 1' },
    { name: 'Test_2_F_Recortado', sex: 'Femenino', race: 'Raza 1', dose: 'Dosis 1'  },
    { name: 'Video 3', sex: 'Masculino', race: 'Raza 2', dose: 'Sin Dosis'  },
    { name: 'Video 4', sex: 'Femenino', race: 'Raza 2', dose: 'Dosis 2' },
    { name: 'Video 5', sex: 'Masculino', race: 'Raza 3', dose: 'Dosis 1' },
    { name: 'Video 6', sex: 'Femenino', race: 'Raza 3', dose: 'Sin Dosis' },
    // Agrega más datos según sea necesario
];

// Función para cargar la lista de videos
function loadVideoList(filter = '') {
    const videoListContainer = document.getElementById('videoList');

    // Limpia la lista antes de agregar nuevos elementos
    videoListContainer.innerHTML = '';

    const filteredVideos = videosData.filter(video => video.name.toLowerCase().includes(filter.toLowerCase()));

    if (filteredVideos.length === 0) {
        // No hay resultados, mostrar mensaje
        const noResultsTitle = document.createElement('h2');
        noResultsTitle.textContent = 'No hay resultados disponibles';
        videoListContainer.appendChild(noResultsTitle);
    } else {
        // Itera sobre los datos y crea las filas de videos
        filteredVideos.forEach(video => {
            const videoItem = document.createElement('div');
            videoItem.classList.add('video-item');

            // Icono de video
            const videoIcon = document.createElement('div');
            videoIcon.classList.add('video-icon');
            videoIcon.innerHTML = '<i class="fas fa-video"></i>'; // Icono de video

            // Nombre del video
            const videoName = document.createElement('div');
            videoName.textContent = video.name;

            // Sexo
            const videoSex = document.createElement('div');
            videoSex.textContent = video.sex;

            // Raza
            const videoRace = document.createElement('div');
            videoRace.textContent = video.race || 'N/A';

            // Dosis
            const videoDose = document.createElement('div');
            videoDose.textContent = video.dose || 'N/A';

            // Acciones
            const actionsContainer = document.createElement('div');
            actionsContainer.classList.add('actions-container');

            // Botón "Ver Resultados" con ícono de ojo
            const viewResultsBtn = document.createElement('button');
            viewResultsBtn.innerHTML = '<i class="fas fa-eye"></i>'; // Ícono de un ojo
            viewResultsBtn.classList.add('view-results-button');
            viewResultsBtn.addEventListener('click', () => viewResults(video.name));

            // Botón "Eliminar"
            const deleteBtn = document.createElement('button');
            deleteBtn.innerHTML = '<i class="fas fa-trash"></i>';
            deleteBtn.classList.add('delete-button');
            deleteBtn.addEventListener('click', () => deleteVideo(video.name));

            // Añadir los botones al contenedor de acciones
            actionsContainer.appendChild(viewResultsBtn);
            actionsContainer.appendChild(deleteBtn);

            // Agrega los elementos a la fila de video
            videoItem.appendChild(videoIcon);
            videoItem.appendChild(videoName);
            videoItem.appendChild(videoSex);
            videoItem.appendChild(videoRace);
            videoItem.appendChild(videoDose);
            videoItem.appendChild(actionsContainer);

            // Agrega la fila de video al contenedor
            videoListContainer.appendChild(videoItem);
        });
    }
}

// Función para filtrar videos según el cuadro de búsqueda
function filterVideos() {
    const searchTerm = document.getElementById('searchBar').value;
    loadVideoList(searchTerm);
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
    loadVideoList();
  };
