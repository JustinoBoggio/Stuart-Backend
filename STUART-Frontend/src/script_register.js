// script_register.js

// Simulamos un listado de correos ya registrados
let registeredUsers = ["usuario@ejemplo.com", "otro@ejemplo.com"];

// Manejamos el submit del formulario de registro
document.getElementById('register-form').addEventListener('submit', function(event) {
  event.preventDefault();

  const email = document.getElementById('register-email').value;
  const pass = document.getElementById('register-pass').value;
  const pass2 = document.getElementById('register-pass2').value;

  // Verifica si las contraseñas coinciden
  if (pass !== pass2) {
    document.getElementById('pass-mismatch-modal').style.display = 'flex';
    return;
  }

  // Verifica si el correo ya está registrado
  if (registeredUsers.includes(email)) {
    document.getElementById('email-exists-modal').style.display = 'flex';
    return;
  }

  // Simulación de registro
  try {
    // (En un caso real, enviarías estos datos al servidor via fetch/AJAX)
    
    // Suponemos que se registra correctamente
    registeredUsers.push(email);

    // Muestra modal de éxito
    document.getElementById('success-modal').style.display = 'flex';
  } catch (error) {
    // Si algo sale mal
    document.getElementById('error-register-modal').style.display = 'flex';
  }
});

// Cierra un modal específico
function closeModal(modalId) {
  document.getElementById(modalId).style.display = 'none';
}

// Redirige al login al hacer click en el botón de “Iniciar Sesión”
function goToLogin() {
  window.location.href = "login.html";
}