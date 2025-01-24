document.addEventListener('DOMContentLoaded', () => {
  fetch('header.html')
      .then(response => response.text())
      .then(data => {
          document.getElementById('header-container').innerHTML = data;

          // Asignar eventos para las imágenes del header
          document.querySelectorAll('.logo-container img').forEach(img => {
              img.addEventListener('click', () => {
                  top.location.href = 'index.html';
              });
          });
      });
});

function redirectToIndex() {
  if (window.location.pathname !== '/index.html') {
      window.location.href = 'index.html';
  }
}

window.redirectToAnotherPage = function (pageName) {
  // Cambia 'otraPagina.html' por la URL de la página a la que deseas redirigir
  window.location.href = pageName;
};