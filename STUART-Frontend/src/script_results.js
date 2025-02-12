// Obtener el nombre del video desde la URL
const urlParams = new URLSearchParams(window.location.search);
const videoName = urlParams.get('video');

// Asignar el nombre del video en la interfaz
document.getElementById("video-name").textContent = videoName;

// Lista de partes del ratón
const bodyParts = [
  "Nariz",
  "Oreja Derecha",
  "Oreja Izquierda",
  "Nuca",
  "Columna Media",
  "Base Cola",
  "Mitad Cola",
  "Final Cola",
];

// Índice actual del carrusel
let currentIndex = 0;

// Referencias a elementos HTML
const carousel = document.querySelector(".carousel");
const dataContainer = document.querySelector(".data-container");

function updateCarousel() {
  const carouselItems = document.querySelectorAll(".carousel-item");

  carouselItems.forEach((item, index) => {
    if (index === currentIndex) {
      item.classList.add("active"); // Añade la clase para el ítem visible
    } else {
      item.classList.remove("active"); // Quita la clase para los ítems no visibles
    }
  });

  // Actualiza los datos dinámicos (tabla, imagen, etc.)
  const part = bodyParts[currentIndex];
  updateData(part);
}


// Avanza en el carrusel
function nextItem() {
  if (currentIndex < bodyParts.length - 1) {
    currentIndex++;
    updateCarousel();
  }
}

// Retrocede en el carrusel
function prevItem() {
  if (currentIndex > 0) {
    currentIndex--;
    updateCarousel();
  }
}

// Actualiza la tabla y los campos según la parte seleccionada
async function updateData(bodyPart) {
  try {
      const response = await fetch(`http://127.0.0.1:5000/get_results/${encodeURIComponent(videoName)}/${encodeURIComponent(bodyPart)}`);
      const data = await response.json();

      console.log("Datos recibidos:", data); // Debug: Ver JSON devuelto por el backend

      if (!data || data.error) {
          console.error("Error en la respuesta del backend:", data ? data.error : "Respuesta vacía");
          return;
      }

      // Actualizar la tabla de tiempos de curiosidad
      const tableBody = document.querySelector(".table-container tbody");
      tableBody.innerHTML = "";

      if (data.times && data.times.length > 0) {
          data.times.forEach(entry => {
              const row = document.createElement("tr");
              row.innerHTML = `
                  <td>${videoName}</td>
                  <td>${bodyPart}</td>
                  <td>${entry.object}</td>
                  <td>${entry.time.toFixed(2)} segundos</td>
              `;
              tableBody.appendChild(row);
          });

          const totalTime = data.times.reduce((sum, entry) => sum + entry.time, 0).toFixed(2);
          document.querySelector(".table-summary").textContent = `Tiempo total: ${totalTime} segundos`;
      } else {
          document.querySelector(".table-summary").textContent = `Tiempo total: 0.00 segundos`;
      }

      document.getElementById("total-distance").value = data.distance ? data.distance.toFixed(2) : "0.00";
      document.getElementById("central-distance").value = data.central_distance ? data.central_distance.toFixed(2) : "0.00";
      document.getElementById("central-time").value = data.central_time ? data.central_time.toFixed(2) : "0.00";
      document.getElementById("central-entries").value = data.central_entries || "0";
      document.getElementById("central-exits").value = data.central_exits || "0";

      // Obtener imagen del mapa de trayectoria
      await fetchTrajectoryImage(bodyPart);

  } catch (error) {
      console.error("Error al obtener los datos:", error);
  }
}

// Función para obtener la imagen de trayectoria desde el backend
async function fetchTrajectoryImage(part) {
  try {
      // Asegurar que la imagen se obtiene del backend y no de una ruta local
      const imgResponse = await fetch(`http://127.0.0.1:5000/get_trayectoria_image/${encodeURIComponent(videoName)}/${encodeURIComponent(part)}`);

      if (!imgResponse.ok) {
          throw new Error(`Error HTTP: ${imgResponse.status}`);
      }

      const imgBlob = await imgResponse.blob();
      const imgUrl = URL.createObjectURL(imgBlob);

      const mapImageElement = document.getElementById("map-image");

      if (!mapImageElement) {
          console.error("Error: Elemento con id 'map-image' no encontrado en el HTML.");
          return;
      }

      mapImageElement.src = imgUrl;
  } catch (error) {
      console.error("Error al obtener la imagen de trayectoria:", error);
  }
}

// Inicialización
function initialize() {
  // Carga las partes del carrusel
  const carouselItems = bodyParts
    .map((part) => `<div class="carousel-item">${part}</div>`)
    .join("");
  carousel.innerHTML = carouselItems;

  // Actualiza el carrusel por primera vez
  updateCarousel();
}

// Eventos de flechas
document.querySelector(".arrow.left").addEventListener("click", prevItem);
document.querySelector(".arrow.right").addEventListener("click", nextItem);

// Llama a la función de inicialización al cargar completamente el DOM
document.addEventListener('DOMContentLoaded', (event) => {
  initialize();
});