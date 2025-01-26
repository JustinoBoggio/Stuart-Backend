// Llama a la función cuando se carga la página
window.onload = function () {
  getCurrentDate()
  simulateData();
};


// Función para obtener la fecha actual en formato DD/MM/YYYY
function getCurrentDate() {
    const today = new Date();
    const day = today.getDate();
    const month = today.getMonth() + 1; // Los meses comienzan desde 0
    const year = today.getFullYear();

    // Actualizar el contenido del elemento con el ID "current-date" con la fecha actual
    document.getElementById('current-date').textContent = `${day}/${month}/${year}`;
  }


function simulateData() {
  // Simular datos que provienen de un sistema externo
  var simulatedData = {
    raza: 'Labrador',
    sexo: 'Macho',
    tiempoTotal: '5:00 min',
    distanciaRecorrida: '244 m',
    tiempoReposo: '1:52 min',
    tiempoCuriosidad: '3:12 min',
  };

  // Mostrar los datos en los campos correspondientes
  document.getElementById('raza').textContent = simulatedData.raza;
  document.getElementById('sexo').textContent = simulatedData.sexo;
  document.getElementById('tiempo-total').textContent = simulatedData.tiempoTotal;
  document.getElementById('distancia-recorrida').textContent = simulatedData.distanciaRecorrida;
  document.getElementById('tiempo-reposo').textContent = simulatedData.tiempoReposo;
  document.getElementById('tiempo-curiosidad').textContent = simulatedData.tiempoCuriosidad;

}

function mostrarModal(modalId) {
  var modal = document.getElementById(modalId);
  // Obtener la fuente desde los parámetros de la URL
  const urlParams = new URLSearchParams(window.location.search);
  var source = urlParams.get('source');
  //const source = urlParams.get('source');
  console.log('source:', source);
  console.log('modalId:', modalId);
  if(modalId == "descartarModal" && source == "library"){
    volverAlMenuPrincipal();
  }else{
    modal.style.display = 'flex';
  }
  
}

function cerrarModal(modalId) {
  var modal = document.getElementById(modalId);
  modal.style.display = 'none';
}

function descartarResultados() {
  // Lógica para descartar resultados
  console.log('Resultados descartados');
  cerrarModal('descartarModal');
  window.location.href = 'index.html';
}

function subirOtroVideo() {
  // Lógica para redirigir a new_vid.html
  window.location.href = 'new_vid.html';
}

function volverAlMenuPrincipal() {
  // Lógica para redirigir a index.html
  window.location.href = 'index.html';
}