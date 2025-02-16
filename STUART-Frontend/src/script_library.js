// Función para cargar la lista de videos
async function loadVideoList(filter = '') {
    const videoListContainer = document.getElementById('videoList');

    // Limpia la lista antes de agregar nuevos elementos
    videoListContainer.innerHTML = '';
    try {
        
        const response = await fetch('http://localhost:5000/get_videos');
        const videosData = await response.json();
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

                // Cantidad
                const videoAmount = document.createElement('div');
                if (video.amount === null || video.amount === undefined || video.amount == '') {
                    videoAmount.textContent = 'N/A'
                }
                else{
                    videoAmount.textContent = video.amount + ' ml' ;
                }
    
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
                videoItem.appendChild(videoAmount);
                videoItem.appendChild(actionsContainer);
    
                // Agrega la fila de video al contenedor
                videoListContainer.appendChild(videoItem);
            });
        }

    }
    catch (error) {
        console.error("Error al obtener la lista de videos:", error);
    }
}

// Función para filtrar videos según el cuadro de búsqueda
function filterVideos() {
    const searchTerm = document.getElementById('searchBar').value;
    loadVideoList(searchTerm);
}


// Función para ver resultados
function viewResults(videoName) {
    window.location.href = `results.html?video=${encodeURIComponent(videoName)}`;
}

// Función para eliminar video 
function deleteVideo(videoName) {
    const modal = document.getElementById("modalOverlay");
    const modalText = document.getElementById("modalText");
    const confirmBtn = document.getElementById("confirmBtn");
    const cancelBtn = document.getElementById("cancelBtn");

    if (!modal) {
        console.error("No se encontró el modal en el HTML.");
        return;
    }

    // Mostrar el modal con el nombre del video
    modalText.textContent = `¿Estás seguro de eliminar el video: ${videoName}?`;
    modal.classList.remove("hidden");

    // Evento de confirmación
    confirmBtn.onclick = async () => {
        try {
            const response = await fetch(`http://localhost:5000/delete_video/${encodeURIComponent(videoName)}`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`Error HTTP ${response.status}`);
            }

            const result = await response.json();

            if (result.status === "success") {
                loadVideoList(); // Recargar la lista de videos
                console.log("Video eliminado con éxito");
            } else {
                console.error("Error al eliminar el video:", result.message);
            }
        } catch (error) {
            console.error("Error al eliminar el video:", error);
        }

        modal.classList.add("hidden"); // Ocultar el modal después de eliminar
    };

    // Evento para cancelar la eliminación
    cancelBtn.onclick = () => {
        modal.classList.add("hidden"); // Ocultar modal sin eliminar
    };
}


// Función para retroceder (puede ser personalizada según tu navegación)
function goBack() {
    window.history.back();
}

// Llama a la función para cargar la lista de videos al cargar la página
window.onload = function () {
    loadVideoList();
    document.getElementById("modalOverlay").classList.add("hidden"); // Ocultar modal por defecto
};
