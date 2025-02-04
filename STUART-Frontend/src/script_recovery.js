// recovery.js

// Variables simulando la existencia de usuarios
const registeredEmails = ["usuario@ejemplo.com", "otro@ejemplo.com"];

// Almacena el código generado y el temporizador
let generatedCode = "";
let countdown;
let timeLeft = 300; // 5 minutos en segundos

// Maneja el submit del formulario
document.getElementById("recovery-form").addEventListener("submit", function(event) {
  event.preventDefault();

  const email = document.getElementById("recovery-email").value;

  // Verifica si el email está en la base de datos simulada
  if (!registeredEmails.includes(email)) {
    // Muestra el modal de advertencia
    document.getElementById("warning-modal").style.display = "flex";
  } else {
    // Genera el código de 6 dígitos
    generatedCode = generateCode(6);

    // Aquí se enviaría el correo con la librería adecuada
    // simulación: console.log(`Enviando código a ${email}: ${generatedCode}`);
    alert(`(Simulación) Código enviado a ${email}: ${generatedCode}`);

    // Muestra la ventana popUp de código
    openCodePopup();
  }
});

// Cierra modal genérico
function closeModal(modalId) {
  document.getElementById(modalId).style.display = "none";
}

// Genera un código aleatorio de 'length' dígitos
function generateCode(length) {
  let code = "";
  for (let i = 0; i < length; i++) {
    code += Math.floor(Math.random() * 10); // dígito aleatorio [0..9]
  }
  return code;
}

// Abre el popUp de código y arranca el temporizador
function openCodePopup() {
  document.getElementById("code-popup").style.display = "flex";
  startCountdown();
}

// Inicia el conteo regresivo
function startCountdown() {
  const countdownTimer = document.getElementById("countdown-timer");

  countdown = setInterval(() => {
    if (timeLeft <= 0) {
      clearInterval(countdown);
      // Cierra popUp y obliga a reenviar código
      //document.getElementById("code-popup").style.display = "none";
      //alert("Se acabó el tiempo, por favor solicita un nuevo código.");
      document.getElementById("error-time-ended-modal").style.display = "flex";
    } else {
      // Calcula minutos y segundos
      const minutes = Math.floor(timeLeft / 60);
      const seconds = timeLeft % 60;
      countdownTimer.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
      timeLeft--;
    }
  }, 1000);
}

// Verifica el código ingresado
function verifyCode() {
  // Toma los 6 inputs
  const digits = document.querySelectorAll(".code-digit");
  let enteredCode = "";
  digits.forEach((digit) => (enteredCode += digit.value));

  if (enteredCode === generatedCode) {
    // Limpia el temporizador y cierra popUp
    clearInterval(countdown);
    document.getElementById("code-popup").style.display = "none";

    // Muestra popUp de nueva contraseña
    document.getElementById("new-pass-popup").style.display = "flex";
  } else {
    // Muestra modal de error
    document.getElementById("error-code-modal").style.display = "flex";
  }
}

// Actualiza la contraseña
function updatePassword() {
  const newPass = document.getElementById("newPass").value;
  const confirmNewPass = document.getElementById("confirmNewPass").value;

  if (newPass === confirmNewPass && newPass !== "") {
    
    // Almacenar las contraseñas en BD

    document.getElementById('success-modal').style.display = 'flex';

  } else {
    // Muestra modal de error para contraseñas
    document.getElementById("error-modal").style.display = "flex";
  }
}

// Cierra modal de éxito
function closeModalSuccess(modalId) {
    document.getElementById(modalId).style.display = "none";

    // Cierra el popUp y redirige al login
    document.getElementById("new-pass-popup").style.display = "none";

    // Redirige a la página de LogIn
    window.location.href = "login.html";
}

// Cierra modal de Tiempo Finalizado
function closeModalTimenEnded(modalId) {
    document.getElementById(modalId).style.display = "none";

    // Cierra el popUp de Ingreso de Código
    document.getElementById("code-popup").style.display = "none";
}

// Abre el modal con mensaje personalizado
// function openModal(action) {
//     currentAction = action;
//     const modal = document.getElementById('error-modal');
//     const modalMessage = document.getElementById('modal-message');

//     // Personaliza el mensaje
//     if (action === "ErrorCode") {
//         modalMessage.innerText = "El Código ingresado es incorrecto.";
//     } else if (action === "ErrorPassword") {
//         modalMessage.innerText = "Las contraseñas ingresadas no coinciden.";
//     } else if (action === "ErrorTimeEnded") {
//         modalMessage.innerText = "Se acabó el tiempo, por favor solicita un nuevo código.";
//     }

//     modalInput.value = ""; // Limpia el campo de entrada
//     modal.style.display = "flex" // Muestra el modal
// }
  