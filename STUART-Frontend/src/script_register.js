document.getElementById('register-form').addEventListener('submit', async function(event) {
  event.preventDefault();

  const email = document.getElementById('register-email').value;
  const pass = document.getElementById('register-pass').value;
  const pass2 = document.getElementById('register-pass2').value;

  // Verifica si las contraseñas coinciden
  if (pass !== pass2) {
    document.getElementById('pass-mismatch-modal').style.display = 'flex';
    return;
  }

  try {
    const response = await fetch('http://localhost:5000/api/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ email, password: pass })
    });

    const result = await response.json();

    if (response.ok) {
      // Registro exitoso
      document.getElementById('success-modal').style.display = 'flex';
    } else {
      // Maneja errores específicos
      if (result.error === 'El correo ya está registrado') {
        document.getElementById('error-register-modal').querySelector('p').textContent = 'El correo ya está registrado.';
      } else {
        document.getElementById('error-register-modal').querySelector('p').textContent = 'Ha ocurrido un error al registrar un nuevo usuario.';
      }
      document.getElementById('error-register-modal').style.display = 'flex';
    }
  } catch (error) {
    // Si algo sale mal
    document.getElementById('error-register-modal').querySelector('p').textContent = 'Error al registrar. Inténtalo de nuevo.';
    document.getElementById('error-register-modal').style.display = 'flex';
  }
});


// Cierra un modal específico
function closeModal(modalId) {
  document.getElementById(modalId).style.display = 'none';
}

// // Redirige al login al hacer click en el botón de “Iniciar Sesión”
function goToLogin() {
  window.location.href = "index.html";
}