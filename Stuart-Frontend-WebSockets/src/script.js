// Función para obtener la fecha actual en formato DD/MM/YYYY
function getCurrentDate() {
    const today = new Date();
    const day = today.getDate();
    const month = today.getMonth() + 1; // Los meses comienzan desde 0
    const year = today.getFullYear();

    return `${day}/${month}/${year}`;
  }
// Actualizar el contenido del elemento con el ID "current-date" con la fecha actual
document.getElementById('current-date').innerText = getCurrentDate();

function redirectToAnotherPage(pageName) {
  // Cambia 'otraPagina.html' por la URL de la página a la que deseas redirigir
  window.location.href = pageName;
}