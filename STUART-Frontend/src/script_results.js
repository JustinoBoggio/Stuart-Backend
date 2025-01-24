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
function updateData(part) {
  // Actualiza la imagen y el título
  const imageContainer = document.querySelector(".image-container img");
  const imageTitle = document.querySelector(".image-container h2");
  imageContainer.src = `images/Trayectoria_${part.toLowerCase()}_con_distancia.png`;
  imageTitle.textContent = `Mapa de trayectoria ${part}`;

  // Actualiza la tabla (simulación de datos dinámicos)
  const tableBody = document.querySelector(".table-container tbody");
  tableBody.innerHTML = `
    <tr>
      <td>video.mp4</td>
      <td>${part}</td>
      <td>Objeto A</td>
      <td>15.2</td>
    </tr>
    <tr>
      <td>video.mp4</td>
      <td>${part}</td>
      <td>Objeto B</td>
      <td>10.5</td>
    </tr>
  `;

  // Actualiza el resumen
  const summary = document.querySelector(".table-summary");
  summary.textContent = "Tiempo total: 25.7 segundos";

  // Actualiza los campos
  document.getElementById("total-distance").value = "12.5";
  document.getElementById("central-distance").value = "4.3";
  document.getElementById("central-time").value = "8.7";
  document.getElementById("central-entries").value = "3";
  document.getElementById("central-exits").value = "3";
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

// Llama a la función de inicialización
initialize();