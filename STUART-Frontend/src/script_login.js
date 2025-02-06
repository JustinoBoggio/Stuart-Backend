// login.js

// Maneja el submit del formulario
document.getElementById('login-form').addEventListener('submit', function(event) {
    event.preventDefault();
  
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
  
    // Simulación de credenciales correctas
    // En un entorno real, consultaría al servidor
    const validEmail = "usuario@ejemplo.com";
    const validPassword = "123456";
  
    if (email === validEmail && password === validPassword) {
      // Redirige a la página principal
      window.location.href = "index.html";
    } else {
      // Muestra el modal de error
      document.getElementById('error-modal').style.display = 'flex';
    }
  });
  
  // Cierra un modal específico
  function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
  }
  
  // Redirige a la página de recuperación
  function redirectToRecovery() {
    window.location.href = "recovery.html";
  }

  // Redirige a la página de registro
  function redirectToRegister() {
    window.location.href = "register.html";
  }