/***********************************************************
 * 1) OBTENER PARÁMETROS Y REFERENCIAS DEL DOM
 ************************************************************/
const urlParams = new URLSearchParams(window.location.search);
const videoName = urlParams.get('video');
document.getElementById("video-name").textContent = videoName;

// Partes del ratón (carrusel) que renderizamos
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

// Referencias en el DOM
const carousel = document.querySelector(".carousel");
// Aquí asumes que existen .arrow.left y .arrow.right en tu HTML

/***********************************************************
 * 2) FUNCIONES PARA EL CARRUSEL
 ************************************************************/
function updateCarousel() {
  const carouselItems = document.querySelectorAll(".carousel-item");
  carouselItems.forEach((item, index) => {
    if (index === currentIndex) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });
  // Al cambiar el carrusel, llamamos a updateData() para recargar la info
  const part = bodyParts[currentIndex];
  updateData(part);
}

function nextItem() {
  if (currentIndex < bodyParts.length - 1) {
    currentIndex++;
    updateCarousel();
  }
}

function prevItem() {
  if (currentIndex > 0) {
    currentIndex--;
    updateCarousel();
  }
}

function initialize() {
  // Generamos todos los .carousel-item
  const carouselItemsHTML = bodyParts
    .map(part => `<div class="carousel-item">${part}</div>`)
    .join("");
  carousel.innerHTML = carouselItemsHTML;

  // Mostramos la primera parte
  updateCarousel();
}

// Eventos de flechas
document.querySelector(".arrow.left").addEventListener("click", prevItem);
document.querySelector(".arrow.right").addEventListener("click", nextItem);

// Al cargar el DOM, inicializar
document.addEventListener("DOMContentLoaded", () => {
  initialize();
});

/***********************************************************
 * 3) FUNCIONES PARA MOSTRAR DATOS EN PANTALLA (updateData)
 ************************************************************/

/**
 * Llama a tu endpoint /get_results/{video}/{bodyPart}
 * para obtener los datos y llenar la tabla y campos
 */
async function updateData(bodyPart) {
  try {
    const response = await fetch(
      `http://localhost:5000/get_results/${encodeURIComponent(videoName)}/${encodeURIComponent(bodyPart)}`
    );
    const data = await response.json();
    console.log("Datos recibidos:", data);

    if (!data || data.error) {
      console.error("Error en la respuesta del backend:", data?.error || "Vacío");
      return;
    }

    // Llenar la tabla
    const tableBody = document.querySelector(".table-container tbody");
    tableBody.innerHTML = "";
    if (data.times && data.times.length > 0) {
      data.times.forEach(entry => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${videoName}</td>
          <td>Nariz</td>
          <td>${entry.object}</td>
          <td>${entry.time.toFixed(2)} segundos</td>
        `;
        tableBody.appendChild(row);
      });
      const totalTime = data.times.reduce((sum, e) => sum + e.time, 0).toFixed(2);
      document.querySelector(".table-summary").textContent =
        `Tiempo total: ${totalTime} segundos`;
    } else {
      document.querySelector(".table-summary").textContent =
        "Tiempo total: 0.00 segundos";
    }

    // Rellenar los campos
    document.getElementById("total-distance").value =
      data.distance ? data.distance.toFixed(2) : "0.00";
    document.getElementById("central-distance").value =
      data.central_distance ? data.central_distance.toFixed(2) : "0.00";
    document.getElementById("central-time").value =
      data.central_time ? data.central_time.toFixed(2) : "0.00";
    document.getElementById("central-entries").value =
      data.central_entries || "0";
    document.getElementById("central-exits").value =
      data.central_exits || "0";

    // Cargar la imagen para la vista principal
    await fetchTrajectoryImage(bodyPart);

  } catch (error) {
    console.error("Error al obtener los datos:", error);
  }
}

/** 
 * Genera un Blob URL de la imagen de trayectoria 
 * dada la parte y el video
 */
async function fetchTrajectoryBlobURL(video, part) {
  try {
    const imgResponse = await fetch(
      `http://localhost:5000/get_trayectoria_image/${encodeURIComponent(video)}/${encodeURIComponent(part)}`
    );
    if (!imgResponse.ok) {
      throw new Error(`HTTP Error: ${imgResponse.status}`);
    }
    const imgBlob = await imgResponse.blob();
    return URL.createObjectURL(imgBlob);
  } catch (err) {
    console.error("Error obteniendo imagen trayectoria:", err);
    // Fallback local
    return "images/Trayectoria_Base Cola_con_distancia.png";
  }
}

/** 
 * Asigna la imagen (para pantalla principal)
 */
async function fetchTrajectoryImage(bodyPart) {
  try {
    const imgUrl = await fetchTrajectoryBlobURL(videoName, bodyPart);
    const mapImageElement = document.getElementById("map-image");
    if (mapImageElement) {
      mapImageElement.src = imgUrl;
    } else {
      console.warn("No se encontró #map-image en el DOM");
    }
  } catch (error) {
    console.error("Error al asignar la imagen:", error);
  }
}

async function fetchTrajectoryImageBase64(video, part) {
  try {
    const resp = await fetch(
      `http://localhost:5000/get_trayectoria_image_base64/${encodeURIComponent(video)}/${encodeURIComponent(part)}`
    );
    const data = await resp.json();
    if (data?.error) {
      console.error("Error devolviendo imagen base64:", data.error);
      // fallback
      return "images/Trayectoria_Base Cola_con_distancia.png";
    }
    // Retornamos "data:image/png;base64,...."
    return `data:image/png;base64,${data.image_base64}`;
  } catch (error) {
    console.error("Error en fetchTrajectoryImageBase64:", error);
    return "images/Trayectoria_Base Cola_con_distancia.png";
  }
}

/***********************************************************
 * 4) FUNCIONES PARA DESCARGAR PDF (REPORTE)
 ************************************************************/

// (A) Pedir datos para una parte (reutilizamos lógica parecida a updateData)
async function fetchDataForPart(part) {
  try {
    const response = await fetch(
      `http://localhost:5000/get_results/${encodeURIComponent(videoName)}/${encodeURIComponent(part)}`
    );
    const data = await response.json();
    if (data?.error) {
      console.error("Error en backend:", data.error);
      return null;
    }
    return data;
  } catch (error) {
    console.error("Error fetchDataForPart:", error);
    return null;
  }
}

async function fetchDataVideo() {
  try {
    const response = await fetch(
      `http://localhost:5000/get_information/${encodeURIComponent(videoName)}`, {
        method: 'GET',
        credentials: 'include'  // <-- Importante
      });
    const data = await response.json();
    if (data?.error) {
      console.error("Error en backend:", data.error);
      return null;
    }
    return data;
  } catch (error) {
    console.error("Error fetchDataVideo:", error);
    return null;
  }
}

// (B) Construir tabla en HTML
function buildCuriosityTable(times, part) {
  if (!times || times.length === 0) {
    return `
      <table>
        <thead>
          <tr>
            <th>Nombre Video</th>
            <th>Keypoint</th>
            <th>Objeto</th>
            <th>Tiempo Curiosidad</th>
          </tr>
        </thead>
        <tbody>
          <tr><td colspan="4">Sin datos</td></tr>
        </tbody>
      </table>
    `;
  }
  let rows = "";
  times.forEach(entry => {
    rows += `
      <tr>
        <td>${videoName}</td>
        <td>Nariz</td>
        <td>${entry.object}</td>
        <td>${entry.time.toFixed(2)} seg</td>
      </tr>
    `;
  });
  return `
    <table>
      <thead>
        <tr>
          <th>Nombre Video</th>
          <th>Keypoint</th>
          <th>Objeto</th>
          <th>Tiempo Curiosidad</th>
        </tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
  `;
}

function calculateTotalTime(times) {
  if (!times || times.length === 0) return 0;
  return times.reduce((sum, e) => sum + e.time, 0).toFixed(2);
}

/** 
 * (C) buildPageForPart: construye el HTML de una “página” PDF 
 */
async function buildPageForPart(data, part) {
  // Armamos la tabla, etc.
  const timesHTML = buildCuriosityTable(data.times, part);
  const totalTime = calculateTotalTime(data.times);

  // [IMPORTANTE] Pedimos la imagen en base64 (no blob)
  const base64Img = await fetchTrajectoryImageBase64(videoName, part);

  return `
    <h1 style="text-align:center;">Resultados: ${videoName} - [${part}]</h1>
    <div style="display:flex; gap:20px; justify-content:space-between;">
      <div style="flex:1;">
        <!-- ... la tabla ... -->
        ${timesHTML}
        <div><b>Tiempo total:</b> ${totalTime} seg</div>
        <!-- ... etc. ... -->
      </div>
      <div style="flex:1; text-align:center;">
        <h2>Mapa de trayectoria</h2>
        <img src="${base64Img}" style="width:400px;" alt="Trayectoria"/>
      </div>
    </div>`;
}

/** 
 * (D) Función principal para DESCARGAR PDF 
 */
async function downloadPDF() {
  try {
    const { jsPDF } = window.jspdf; // Asegúrate de haber incluido la librería jsPDF
    const doc = new jsPDF({
      orientation: "landscape",
      unit: "pt", // points
      format: "letter",
    });

    const informationVideo = await fetchDataVideo();
    // Normalizamos los campos
    const raza = informationVideo?.raza || "";
    const sexo = informationVideo?.sexo || "";
    const dosis = informationVideo?.dosis || "";
    // Si "cantidad" está vacía o nula => "N/A", si no => "val ml"
    const cantidadRaw = informationVideo?.cantidad ?? "";
    const cantidadText = cantidadRaw ? `${cantidadRaw} ml` : "N/A";

    const user = informationVideo?.usuario || "";
    const fecha = informationVideo?.fecha || "";
    const hora = informationVideo?.hora || "";

    for (let i = 0; i < bodyParts.length; i++) {
      const part = bodyParts[i];
      const data = await fetchDataForPart(part);
      if (!data) continue;

      // Imagen en Base64 (para el PDF)
      const base64Img = await fetchTrajectoryImageBase64(videoName, part);

      // Nueva página si no es la primera
      if (i > 0) doc.addPage();

      // ======= 1) Título Principal (grande y en bold) =======
      doc.setFont("helvetica", "bold");
      doc.setFontSize(16);
      doc.text(`Resultados: ${videoName} - [${part}]`, 40, 40);

      // ====== [B] Subtítulo “Información del video” ======
      // Ocupa todo el ancho => lo centramos en la página (letter ~ 792 px de ancho, 612 en horizontal).
      // Podemos usar doc.internal.pageSize.width para el ancho total, y /2 para la mitad, con { align: "center" }.
      const pageWidth = doc.internal.pageSize.width;
      doc.setFontSize(14);
      doc.setFont("helvetica", "bold");
      doc.text("Información del Video", pageWidth / 2, 80, { align: "center" });

      // 3) Campos Raza, Dosis, Cantidad en horizontal
      //   (mantemos el mismo Y, pero distintos X)
      const infoY = 110; // la altura donde los situamos
      doc.setFont("helvetica", "normal");
      doc.setFontSize(12);

      doc.text(`Raza del ratón: ${raza}`, 40, infoY);     // Izq
      doc.text(`Sexo: ${sexo}`, 300, infoY); 
      doc.text(`Dosis: ${dosis}`, 480, infoY);           // un poco más a la derecha
      doc.text(`Cantidad: ${cantidadText}`, 640, infoY); // más a la derecha

      // Dejamos un espacio adicional debajo
      let nextY = infoY + 70;

      // ======= 2) Subtítulo “Tiempos de Curiosidad” =======
      doc.setFontSize(14); // Subtítulo (manténlo igual para “Mapa de trayectoria”)
      doc.setFont("helvetica", "bold");
      // Ajusta la Y para bajarlo más (ej. 100) si quieres todavía más espacio
      doc.text("Tiempos de Curiosidad", 40, nextY);

      // ======= 3) Generar la tabla con AutoTable “más abajo” =======
      let tableY = nextY + 40;
      let bodyRows = [];
      if (data.times && data.times.length > 0) {
        bodyRows = data.times.map((entry) => [
          videoName,
          'Nariz',
          entry.object,
          `${entry.time.toFixed(2)} segundos`,
        ]);
      }

      doc.autoTable({
        startY: tableY, // Ajusta para que quede más abajo
        margin: { left: 40, top: 50 },
        tableWidth: 300, // Ancho de la tabla
        theme: "grid",
        head: [
          [
            "Nombre Video",
            "Keypoint",
            "Objeto de interés",
            "Tiempo de Curiosidad",
          ],
        ],
        body: bodyRows,
        headStyles: {
          fillColor: [41, 128, 185],
          textColor: 255,
          fontStyle: "bold",
        },
        styles: {
          font: "helvetica",
          fontSize: 10,
          cellPadding: 5,
        },
      });

      // ======= 4) Campos debajo de la tabla (en negrita) =======
      let finalY = doc.lastAutoTable.finalY + 40;
      const totalTime = (data.times || []).reduce((acc, e) => acc + e.time, 0).toFixed(2);

      // Ponemos en negrita
      doc.setFont("helvetica", "bold");
      doc.setFontSize(11);

      const leftX = 40;
      doc.text(`Tiempo total: ${totalTime} seg`, leftX, finalY);
      finalY += 14;
      doc.text(`Distancia Total Recorrida: ${data.distance?.toFixed(2) || "0.00"} m`, leftX, finalY);
      finalY += 14;
      doc.text(`Distancia Total Recorrida en Área Central: ${data.central_distance?.toFixed(2) || "0.00"} m`, leftX, finalY);
      finalY += 14;
      doc.text(`Tiempo dentro de Área Central: ${data.central_time?.toFixed(2) || "0.00"} seg`, leftX, finalY);
      finalY += 14;
      doc.text(`N° de entradas Área Central: ${data.central_entries || "0"}`, leftX, finalY);
      finalY += 14;
      doc.text(`N° de salidas Área Central: ${data.central_exits || "0"}`, leftX, finalY);

      // ======= 5) Subtítulo e imagen a la derecha =======
      doc.setFontSize(14); // Mismo tamaño que “Tiempos de Curiosidad”
      doc.setFont("helvetica", "bold");
      doc.text("Mapa de trayectoria", 400, nextY);
      // Insertar la imagen base64 en X=400, Y=80, ancho=300, alto=300 (ajusta a gusto)
      doc.addImage(base64Img, "PNG", 360, nextY + 5 , 400, 400);
    }

    const totalPages = doc.getNumberOfPages();
    for (let page = 1; page <= totalPages; page++) {
      doc.setPage(page); // situarnos en esa página

      const pageWidth = doc.internal.pageSize.width;
      const pageHeight = doc.internal.pageSize.height;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(10);

      // Texto horizontal con reporte y fecha/hora
      doc.text(
        `Reporte generado por: ${user}   -   Fecha: ${fecha}   -   Hora: ${hora}`,
        pageWidth / 2,
        pageHeight - 30,
        { align: "center" }
      );

      // Número de página centrado debajo
      doc.text(
        `Página ${page} de ${totalPages}`,
        pageWidth / 2,
        pageHeight - 15,
        { align: "center" }
      );
    }

    // 10) Guardar
    doc.save(`Reporte_${videoName}.pdf`);
  } catch (err) {
    console.error("Error generando PDF con jsPDF y AutoTable:", err);
  }
}