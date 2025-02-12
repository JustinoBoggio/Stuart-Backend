  // login.js

  document.getElementById('login-form').addEventListener('submit', async function(event) {
    event.preventDefault();
  
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
  
    try {
      const response = await fetch('http://localhost:5000/api/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email, password }),
        credentials: 'include' // Esto es crucial para enviar cookies
      });
  
      const result = await response.json();
  
      if (response.ok) {
        // Inicio de sesión exitoso
        window.location.href = "main.html";
      } else {
        // Muestra el modal de error con el mensaje del servidor
        document.getElementById('error-modal').querySelector('p').textContent = result.error;
        document.getElementById('error-modal').style.display = 'flex';
      }
    } catch (error) {
      // Si algo sale mal
      document.getElementById('error-modal').querySelector('p').textContent = 'Error al iniciar sesión. Inténtalo de nuevo.';
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