document.addEventListener('DOMContentLoaded', () => {
  fetch('header.html')
      .then(response => response.text())
      .then(data => {
          document.getElementById('header-container').innerHTML = data;

          // Asignar eventos para las imágenes del header
          document.querySelectorAll('.logo-container img').forEach(img => {
              img.addEventListener('click', () => {
                  top.location.href = 'main.html';
              });
          });
      });
});

function redirectToIndex() {
  if (window.location.pathname !== '/main.html') {
      window.location.href = 'main.html';
  }
}

window.redirectToAnotherPage = function (pageName) {
  // Cambia 'otraPagina.html' por la URL de la página a la que deseas redirigir
  window.location.href = pageName;
};